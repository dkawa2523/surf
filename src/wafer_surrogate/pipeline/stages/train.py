from __future__ import annotations

import csv
import inspect
import json
import random
import re
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Protocol

from wafer_surrogate.core import (
    assess_feature_ood,
    assert_feature_contract_compatible,
    load_feature_contract,
    normalize_feature_contract,
)
from wafer_surrogate.data.h5_dataset import load_narrow_band_dataset
from wafer_surrogate.data.io import NarrowBandDataset
from wafer_surrogate.inference.ood import assess_ood
from wafer_surrogate.models import make_model
from wafer_surrogate.models.api import MODEL_REGISTRY, resolve_model_alias
from wafer_surrogate.runtime import detect_runtime_capabilities
from wafer_surrogate.training import OptionalSparseDependencyUnavailable, SparseDistillConfig, train_sparse_distill
from wafer_surrogate.pipeline.types import ArtifactRef, StageResult
from wafer_surrogate.pipeline.utils import write_csv, write_json
from wafer_surrogate.viz.plots import render_train_output_visuals
from wafer_surrogate.viz.utils import resolve_visualization_config, viz_enabled


MetricFn = Callable[[list[float], list[float]], float]


def _metric_registry() -> dict[str, MetricFn]:
    return {
        "mae": lambda y_true, y_pred: fmean(abs(float(p) - float(t)) for t, p in zip(y_true, y_pred)),
        "rmse": lambda y_true, y_pred: (
            fmean((float(p) - float(t)) ** 2 for t, p in zip(y_true, y_pred)) ** 0.5
        ),
    }


def _score_summary(values: list[float], *, default_threshold: float) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "min": 0.0,
            "mean": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "threshold": float(default_threshold),
        }
    sorted_values = sorted(float(v) for v in values)

    def _pick(q: float) -> float:
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        pos = q * float(len(sorted_values) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(sorted_values) - 1)
        frac = pos - float(lo)
        return ((1.0 - frac) * float(sorted_values[lo])) + (frac * float(sorted_values[hi]))

    p95 = _pick(0.95)
    p99 = _pick(0.99)
    return {
        "count": float(len(sorted_values)),
        "min": float(sorted_values[0]),
        "mean": float(fmean(sorted_values)),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(sorted_values[-1]),
        "threshold": float(max(float(default_threshold), float(p99))),
    }


def _build_dual_ood_reference(
    *,
    reference_conditions: list[dict[str, float]],
    reference_features: list[dict[str, float]],
    default_threshold: float = 3.0,
    max_samples: int = 200,
) -> dict[str, Any]:
    cond_ref = [dict(row) for row in reference_conditions if isinstance(row, Mapping)]
    feat_ref = [dict(row) for row in reference_features if isinstance(row, Mapping)]
    if len(cond_ref) > max_samples:
        cond_ref = cond_ref[:max_samples]
    if len(feat_ref) > max_samples:
        feat_ref = feat_ref[:max_samples]

    cond_scores: list[float] = []
    if cond_ref:
        for row in cond_ref:
            score = assess_ood(
                query_conditions=row,
                reference_conditions=cond_ref,
                threshold=max(9_999.0, float(default_threshold)),
            ).get("distance")
            if isinstance(score, (int, float)):
                cond_scores.append(float(score))

    feat_scores: list[float] = []
    if feat_ref:
        for row in feat_ref:
            score = assess_feature_ood(
                query_features=row,
                reference_features=feat_ref,
                threshold=max(9_999.0, float(default_threshold)),
            ).get("distance")
            if isinstance(score, (int, float)):
                feat_scores.append(float(score))

    condition_summary = _score_summary(cond_scores, default_threshold=default_threshold)
    feature_summary = _score_summary(feat_scores, default_threshold=default_threshold)
    return {
        "schema_version": "1",
        "condition": condition_summary,
        "feature": feature_summary,
        "num_reference_conditions": len(cond_ref),
        "num_reference_features": len(feat_ref),
    }


def _condition_rows_from_features(features: list[dict[str, float]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for row in features:
        cond: dict[str, float] = {}
        for key, value in row.items():
            name = str(key)
            if name.startswith("feat_"):
                cond[name[5:]] = float(value)
            elif name.startswith("recipe_"):
                cond[name] = float(value)
        if cond:
            out.append(cond)
    return out


class TrainBackend(Protocol):
    name: str

    def run(
        self,
        stage: "TrainStage",
        *,
        runtime: Any,
        stage_dirs: dict[str, Path],
        params: dict[str, Any],
        external_inputs: Mapping[str, str],
        input_refs: dict[str, str],
        warnings: list[str],
    ) -> StageResult:
        ...


class TabularTrainBackend:
    name = "tabular"

    def run(
        self,
        stage: "TrainStage",
        *,
        runtime: Any,
        stage_dirs: dict[str, Path],
        params: dict[str, Any],
        external_inputs: Mapping[str, str],
        input_refs: dict[str, str],
        warnings: list[str],
    ) -> StageResult:
        mode_label = str(params.get("mode", self.name)).strip().lower() or self.name
        return stage._run_tabular(
            runtime,
            stage_dirs,
            params,
            external_inputs,
            input_refs,
            warnings,
            forced_model_name=None,
            mode_label=mode_label,
        )


class SparseDistillTrainBackend:
    name = "sparse_distill"

    def run(
        self,
        stage: "TrainStage",
        *,
        runtime: Any,
        stage_dirs: dict[str, Path],
        params: dict[str, Any],
        external_inputs: Mapping[str, str],
        input_refs: dict[str, str],
        warnings: list[str],
    ) -> StageResult:
        capabilities = detect_runtime_capabilities()
        backend_ok = bool(capabilities.sparse_backend)
        if not backend_ok:
            fallback_model = str(params.get("fallback_model", "baseline_vn_linear_trainable"))
            fallback_reason = (
                "sparse_distill fallback: optional sparse backend (torch+MinkowskiEngine) unavailable; "
                f"using fallback model '{fallback_model}'"
            )
            warnings.append(fallback_reason)
            warnings.append(f"sparse backend capabilities: {capabilities.missing_summary()}")
            tab_params = dict(params)
            tab_params["model_name"] = fallback_model
            tab_params["model_variants"] = [fallback_model]
            return stage._run_tabular(
                runtime,
                stage_dirs,
                tab_params,
                external_inputs,
                input_refs,
                warnings,
                forced_model_name=fallback_model,
                mode_label="sparse_distill_fallback",
                fallback_reason=fallback_reason,
            )
        return stage._run_sparse_distill(runtime, stage_dirs, params, external_inputs, input_refs, warnings)


class TrainStage:
    name = "train"

    def _resolve_visualization_config(
        self,
        *,
        runtime: Any,
        params: Mapping[str, Any],
        warnings: list[str],
    ) -> dict[str, Any]:
        payload = runtime.payload.get("config")
        run_cfg = payload.get("visualization") if isinstance(payload, Mapping) else {}
        stage_cfg = params.get("visualization") if isinstance(params, Mapping) else {}
        return resolve_visualization_config(
            run_config=run_cfg if isinstance(run_cfg, Mapping) else {},
            stage_config=stage_cfg if isinstance(stage_cfg, Mapping) else {},
            warnings=warnings,
        )

    def _normalize_split_info(self, split_info: Mapping[str, Any], *, loader_mode_default: str) -> dict[str, Any]:
        out = {str(key): value for key, value in split_info.items()}
        out["schema_version"] = 2
        out["num_train_runs"] = int(out.get("num_train_runs", 0))
        out["num_valid_runs"] = int(out.get("num_valid_runs", 0))
        out["leak_checked"] = bool(out.get("leak_checked", False))
        out["reason"] = str(out.get("reason", ""))
        out["loader_mode"] = str(out.get("loader_mode", loader_mode_default) or loader_mode_default)
        required = {"num_train_runs", "num_valid_runs", "leak_checked", "reason", "loader_mode"}
        missing = sorted(required.difference(set(out.keys())))
        if missing:
            raise ValueError(f"train split_info missing required keys: {missing}")
        return out

    def _stage_external_inputs(self, runtime: Any) -> dict[str, str]:
        run_cfg = getattr(runtime, "run_config", None)
        stages = getattr(run_cfg, "stages", [])
        for stage_cfg in stages:
            if str(getattr(stage_cfg, "name", "")) != self.name:
                continue
            raw = getattr(stage_cfg, "external_inputs", {})
            if isinstance(raw, Mapping):
                return {str(key): str(value) for key, value in raw.items()}
        return {}

    def _read_feature_rows(self, path: Path) -> list[dict[str, float]]:
        with path.open("r", encoding="utf-8", newline="") as fp:
            rows = [dict(row) for row in csv.DictReader(fp)]
        return [{str(key): float(value) for key, value in row.items()} for row in rows]

    def _read_target_values(self, path: Path) -> list[float]:
        with path.open("r", encoding="utf-8", newline="") as fp:
            rows = [dict(row) for row in csv.DictReader(fp)]
        if not rows:
            return []
        key = "target" if "target" in rows[0] else next(
            (name for name in rows[0] if name not in {"index", "sample_index"}),
            None,
        )
        if key is None:
            raise ValueError(f"train targets_csv has no usable column: {path}")
        return [float(row[key]) for row in rows]

    def _read_json_mapping(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, dict):
            raise ValueError(f"train preprocess_bundle_json must be mapping: {path}")
        return payload

    def _build_tabular_feature_contract(
        self,
        *,
        features: list[dict[str, float]],
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not features:
            raise ValueError("train contract mismatch (tabular features must be non-empty)")
        feature_names = [str(key) for key in features[0].keys()]
        seen = set(feature_names)
        for row in features[1:]:
            for key in row.keys():
                skey = str(key)
                if skey not in seen:
                    feature_names.append(skey)
                    seen.add(skey)
        recipe_keys = sorted(
            {name[5:] for name in feature_names if name.startswith("feat_")}
        )
        if not recipe_keys:
            recipe_keys = sorted(
                {name for name in feature_names if name.startswith("recipe_")}
            )
        contract = {
            "feature_names": feature_names,
            "feat_dim": len(feature_names),
            "cond_dim": len(recipe_keys),
            "recipe_keys": recipe_keys,
            "band_width": float(params.get("band_width", 0.5)),
            "min_grad_norm": float(params.get("min_grad_norm", 1e-6)),
        }
        return normalize_feature_contract(
            contract,
            source="train.tabular_contract",
            require_extended=True,
        )

    def _train_one(
        self,
        *,
        model_name: str,
        model_kwargs: dict[str, Any],
        features: list[dict[str, float]],
        targets: list[float],
    ) -> tuple[Any, dict[str, float]]:
        kwargs = dict(model_kwargs)
        try:
            factory = MODEL_REGISTRY.get(model_name)
            allowed = set(inspect.signature(factory).parameters.keys())
            kwargs = {key: value for key, value in kwargs.items() if key in allowed}
        except Exception:
            kwargs = dict(model_kwargs)
        model = make_model(model_name, **kwargs)
        model.fit(features, targets)
        preds = [float(model.predict(sample)) for sample in features]
        metrics = {name: fn(targets, preds) for name, fn in _metric_registry().items()}
        metrics["target_mean"] = fmean(targets)
        metrics["prediction_mean"] = fmean(preds)
        return model, metrics

    def _resolve_model_variants(
        self,
        *,
        params: dict[str, Any],
        default_model_kwargs: dict[str, Any],
        forced_model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        if forced_model_name:
            return [
                {
                    "variant_id": "variant_00",
                    "model": str(forced_model_name),
                    "model_kwargs": dict(default_model_kwargs),
                }
            ]

        variants: list[dict[str, Any]] = []
        model_variants_cfg = params.get("model_variants")
        if isinstance(model_variants_cfg, list):
            for index, item in enumerate(model_variants_cfg):
                if isinstance(item, str):
                    variants.append(
                        {
                            "variant_id": f"variant_{index:02d}",
                            "model": str(item),
                            "model_kwargs": dict(default_model_kwargs),
                        }
                    )
                    continue
                if isinstance(item, Mapping):
                    model_name = item.get("model") if item.get("model") is not None else item.get("name")
                    if model_name is None:
                        continue
                    merged_kwargs = dict(default_model_kwargs)
                    raw_kwargs = item.get("model_kwargs") if item.get("model_kwargs") is not None else item.get("kwargs")
                    if isinstance(raw_kwargs, Mapping):
                        merged_kwargs.update({str(key): value for key, value in raw_kwargs.items()})
                    variants.append(
                        {
                            "variant_id": f"variant_{index:02d}",
                            "model": str(model_name),
                            "model_kwargs": merged_kwargs,
                        }
                    )

        if not variants:
            variants = [
                {
                    "variant_id": "variant_00",
                    "model": str(params.get("model_name", "baseline_vn_linear_trainable")),
                    "model_kwargs": dict(default_model_kwargs),
                }
            ]
        return variants

    def _build_cv_folds(
        self,
        *,
        row_refs: list[dict[str, Any]],
        n_samples: int,
        cv_folds: int,
        seed: int,
    ) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
        run_ids: list[str] = []
        for idx in range(n_samples):
            if idx < len(row_refs):
                run_ids.append(str(row_refs[idx].get("run_id", f"row_{idx}")))
            else:
                run_ids.append(f"row_{idx}")
        unique_runs = sorted(set(run_ids))

        if n_samples < 2 or cv_folds < 2:
            return [], [], {
                "schema_version": 2,
                "mode": "disabled",
                "reason": "insufficient_rows_or_folds",
                "cv_folds": 0,
                "requested_folds": int(cv_folds),
                "unique_runs": len(unique_runs),
                "num_train_runs": 0,
                "num_valid_runs": 0,
                "leak_checked": False,
                "seed": int(seed),
                "loader_mode": "in_memory",
            }

        if len(unique_runs) < 2:
            return [], [], {
                "schema_version": 2,
                "mode": "disabled",
                "reason": "insufficient_unique_runs",
                "cv_folds": 0,
                "requested_folds": int(cv_folds),
                "unique_runs": len(unique_runs),
                "num_train_runs": len(unique_runs),
                "num_valid_runs": 0,
                "leak_checked": False,
                "seed": int(seed),
                "loader_mode": "in_memory",
            }

        k = max(2, min(int(cv_folds), len(unique_runs)))
        shuffled = list(unique_runs)
        random.Random(int(seed)).shuffle(shuffled)
        run_folds = [shuffled[i::k] for i in range(k)]

        train_folds: list[list[int]] = []
        valid_folds: list[list[int]] = []
        leak_checked = True
        for fold_idx in range(k):
            valid_runs = set(run_folds[fold_idx])
            train_idx = [idx for idx, run_id in enumerate(run_ids) if run_id not in valid_runs]
            valid_idx = [idx for idx, run_id in enumerate(run_ids) if run_id in valid_runs]
            if not train_idx or not valid_idx:
                continue
            train_run_set = {run_ids[idx] for idx in train_idx}
            valid_run_set = {run_ids[idx] for idx in valid_idx}
            if train_run_set.intersection(valid_run_set):
                leak_checked = False
                raise ValueError("data leakage detected: train/valid run_id overlap")
            train_folds.append(train_idx)
            valid_folds.append(valid_idx)

        split_info = {
            "schema_version": 2,
            "mode": "kfold_by_run",
            "cv_folds": len(valid_folds),
            "requested_folds": int(cv_folds),
            "unique_runs": len(unique_runs),
            "num_train_runs": max(1, len(unique_runs) - max(1, len(run_folds[0]))),
            "num_valid_runs": max(1, len(run_folds[0])),
            "leak_checked": leak_checked,
            "reason": "",
            "seed": int(seed),
            "loader_mode": "in_memory",
        }
        return train_folds, valid_folds, split_info

    def _cross_validate(
        self,
        *,
        model_name: str,
        model_kwargs: dict[str, Any],
        features: list[dict[str, float]],
        targets: list[float],
        train_folds: list[list[int]],
        valid_folds: list[list[int]],
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        if not train_folds or not valid_folds:
            return {}, []

        fold_rows: list[dict[str, Any]] = []
        for fold_idx, (train_idx, valid_idx) in enumerate(zip(train_folds, valid_folds)):
            fold_model, _ = self._train_one(
                model_name=model_name,
                model_kwargs=model_kwargs,
                features=[features[idx] for idx in train_idx],
                targets=[targets[idx] for idx in train_idx],
            )
            y_valid = [targets[idx] for idx in valid_idx]
            pred_valid = [float(fold_model.predict(features[idx])) for idx in valid_idx]
            metric_values = {name: fn(y_valid, pred_valid) for name, fn in _metric_registry().items()}
            fold_rows.append(
                {
                    "fold": int(fold_idx),
                    "num_train": len(train_idx),
                    "num_valid": len(valid_idx),
                    **{f"valid_{name}": float(value) for name, value in metric_values.items()},
                }
            )

        summary = {
            "cv_valid_mae": fmean(float(row["valid_mae"]) for row in fold_rows),
            "cv_valid_rmse": fmean(float(row["valid_rmse"]) for row in fold_rows),
        }
        return summary, fold_rows

    def _load_narrow_band_dataset(self, *, runtime: Any, external_inputs: Mapping[str, str], input_refs: dict[str, str]) -> NarrowBandDataset:
        dataset = runtime.payload.get("narrow_band_dataset")
        if isinstance(dataset, NarrowBandDataset):
            return dataset

        manifest_path = external_inputs.get("narrow_band_manifest_json")
        if manifest_path:
            resolved = Path(manifest_path)
            if not resolved.exists() or not resolved.is_file():
                raise ValueError(f"train narrow_band_manifest_json does not exist: {resolved}")
            with resolved.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            if isinstance(payload, Mapping):
                ds_path = payload.get("dataset_path")
                if ds_path is not None:
                    resolved_ds = Path(str(ds_path))
                    if not resolved_ds.is_absolute():
                        resolved_ds = (resolved.parent / resolved_ds).resolve()
                    external_inputs = dict(external_inputs)
                    if str(resolved_ds).endswith(".h5"):
                        external_inputs["narrow_band_h5"] = str(resolved_ds)
                    elif str(resolved_ds).endswith(".zarr"):
                        external_inputs["narrow_band_zarr"] = str(resolved_ds)
                    else:
                        external_inputs["narrow_band_json"] = str(resolved_ds)
                input_refs["narrow_band_manifest_json"] = str(resolved)

        for key in ("narrow_band_h5", "narrow_band_zarr", "narrow_band_json"):
            raw_path = external_inputs.get(key)
            if raw_path is None:
                continue
            resolved = Path(str(raw_path))
            if not resolved.exists():
                raise ValueError(f"train {key} does not exist: {resolved}")
            if key == "narrow_band_json" and not resolved.is_file():
                raise ValueError(f"train {key} must be a file: {resolved}")
            dataset = load_narrow_band_dataset(resolved)
            input_refs[key] = str(resolved)
            return dataset

        raise ValueError(
            "train sparse_distill requires narrow-band artifact from featurization or external_inputs.narrow_band_h5|narrow_band_zarr|narrow_band_json"
        )

    def _run_sparse_distill(self, runtime: Any, stage_dirs: dict[str, Path], params: dict[str, Any], external_inputs: Mapping[str, str], input_refs: dict[str, str], warnings: list[str]) -> StageResult:
        runtime_dataset = runtime.payload.get("narrow_band_dataset")
        dataset = self._load_narrow_band_dataset(runtime=runtime, external_inputs=external_inputs, input_refs=input_refs)
        external_dataset_used = not isinstance(runtime_dataset, NarrowBandDataset)
        seed = int(params.get("seed", 0))

        point_manifest = runtime.payload.get("point_level_manifest")
        if not isinstance(point_manifest, Mapping):
            point_manifest_path = external_inputs.get("point_level_manifest_json")
            if point_manifest_path:
                resolved = Path(str(point_manifest_path))
                if not resolved.exists() or not resolved.is_file():
                    raise ValueError(f"train point_level_manifest_json does not exist: {resolved}")
                with resolved.open("r", encoding="utf-8") as fp:
                    loaded = json.load(fp)
                if isinstance(loaded, Mapping):
                    point_manifest = loaded
                    input_refs["point_level_manifest_json"] = str(resolved)

        expected_contract: Mapping[str, Any] | None = None
        feature_contract_raw: Mapping[str, Any] = {}
        if isinstance(point_manifest, Mapping):
            maybe_contract = point_manifest.get("feature_contract")
            if isinstance(maybe_contract, Mapping):
                feature_contract_raw = maybe_contract
                expected_contract = normalize_feature_contract(
                    maybe_contract,
                    source="point_level_manifest.feature_contract",
                    require_extended=True,
                )
        feature_contract_path = external_inputs.get("feature_contract_json")
        if feature_contract_path:
            loaded_contract = load_feature_contract(feature_contract_path)
            input_refs["feature_contract_json"] = str(Path(feature_contract_path))
            if expected_contract is not None:
                assert_feature_contract_compatible(
                    expected=expected_contract,
                    actual=loaded_contract,
                    context="train.feature_contract",
                )
            expected_contract = loaded_contract
        recipe_keys = [str(v) for v in feature_contract_raw.get("recipe_keys", [])] if feature_contract_raw else []
        if not recipe_keys and isinstance(expected_contract, Mapping):
            recipe_keys = [str(v) for v in expected_contract.get("recipe_keys", [])]
        if not recipe_keys and dataset.runs:
            recipe_keys = [f"recipe_{idx}" for idx in range(len(dataset.runs[0].recipe))]
        if external_dataset_used and expected_contract is None:
            raise ValueError(
                "contract mismatch (recipe_keys unresolved): external narrow-band input requires feature_contract_json or point_level_manifest_json"
            )
        if external_dataset_used:
            if not recipe_keys or all(re.match(r"^recipe_\d+$", key) for key in recipe_keys):
                raise ValueError("contract mismatch (recipe_keys unresolved)")

        cfg = SparseDistillConfig(
            teacher_epochs=max(1, int(params.get("teacher_epochs", 40))),
            student_epochs=max(1, int(params.get("student_epochs", 60))),
            learning_rate=float(params.get("learning_rate", 1e-3)),
            weight_decay=float(params.get("weight_decay", 1e-6)),
            alpha=float(params.get("distill_alpha", 0.5)),
            beta=float(params.get("distill_beta", 0.2)),
            gamma=float(params.get("distill_gamma", 0.1)),
            batch_size_steps=max(1, int(params.get("batch_size_steps", 8))),
            max_points=max(1, int(params.get("sparse_batch_points", 20000))),
            seed=seed,
            device=str(params.get("device", "cpu")),
            recipe_keys=recipe_keys,
            band_width=float(feature_contract_raw.get("band_width", params.get("band_width", 0.5))),
            min_grad_norm=float(feature_contract_raw.get("min_grad_norm", params.get("min_grad_norm", 1e-6))),
            grad_clip_norm=float(params.get("grad_clip_norm", params.get("grad_clip", 5.0))),
            early_stopping_patience=max(0, int(params.get("early_stopping_patience", 0))),
            lr_scheduler=str(params.get("lr_scheduler", "none")).strip().lower() or "none",
            latent_dim=max(0, int(params.get("latent_dim", 0))),
            rollout_loss_enabled=bool(params.get("rollout_loss_enabled", False)),
            rollout_k=max(1, int(params.get("rollout_k", 3))),
            rollout_weight=float(params.get("rollout_weight", 0.1)),
            temporal_step_weight=str(params.get("temporal_step_weight", "uniform")).strip().lower() or "uniform",
            step_sampling_policy=str(params.get("step_sampling_policy", "uniform")).strip().lower() or "uniform",
            strict_split=bool(params.get("strict_split", False)),
            patch_size=max(0, int(params.get("patch_size", 0))),
            patches_per_step=max(1, int(params.get("patches_per_step", 1))),
            sparse_model_profile=str(params.get("sparse_model_profile", "small")).strip().lower() or "small",
            hidden_channels=int(params.get("hidden_channels", 0)),
            num_blocks=int(params.get("num_blocks", 0)),
            dropout=float(params.get("dropout", 0.0)),
            residual=bool(params.get("residual", True)),
        )

        try:
            result = train_sparse_distill(dataset=dataset, output_dir=stage_dirs["outputs"], config=cfg)
        except OptionalSparseDependencyUnavailable as exc:
            raise ValueError(f"sparse_distill requires optional dependencies: {exc}") from exc
        if expected_contract is not None:
            assert_feature_contract_compatible(
                expected=expected_contract,
                actual=result.feature_contract.to_dict(),
                context="train.sparse_distill_contract",
            )

        split_info = self._normalize_split_info(result.split_info, loader_mode_default="streaming")

        runtime.payload["trained_model"] = result.model
        runtime.payload["trained_model_name"] = "sparse_vn_student"
        runtime.payload["train_metrics"] = dict(result.metrics)
        runtime.payload["train_comparisons"] = [dict(row) for row in result.comparisons]
        runtime.payload["train_metric_rows"] = [dict(row) for row in result.metric_rows]
        runtime.payload["train_split_info"] = dict(split_info)
        runtime.payload["feature_contract"] = result.feature_contract.to_dict()
        runtime.payload["condition_scaler"] = dict(result.condition_scaler)
        model_architecture = dict(getattr(result.model, "architecture", {}))
        sparse_condition_refs: list[dict[str, float]] = []
        sparse_feature_refs: list[dict[str, float]] = []
        max_ref_rows = 200
        for run in dataset.runs:
            cond = {
                str(key): float(value)
                for key, value in zip(result.feature_contract.recipe_keys, run.recipe)
            }
            if cond:
                sparse_condition_refs.append(cond)
            for step_index, step in enumerate(run.steps):
                limit = min(len(step.coords), len(step.feat), max_ref_rows - len(sparse_feature_refs))
                for point_index in range(max(0, limit)):
                    coord = step.coords[point_index]
                    feat = step.feat[point_index]
                    row = {
                        "phi": float(feat[0]) if len(feat) > 0 else 0.0,
                        "grad_norm": float(feat[1]) if len(feat) > 1 else 0.0,
                        "curvature_proxy": float(feat[2]) if len(feat) > 2 else 0.0,
                        "band_distance": float(feat[3]) if len(feat) > 3 else abs(float(feat[0])) if feat else 0.0,
                        "coord_x": float(coord[0]) if len(coord) > 0 else 0.0,
                        "coord_y": float(coord[1]) if len(coord) > 1 else 0.0,
                        "coord_z": float(coord[2]) if len(coord) > 2 else 0.0,
                        "step_index": float(step_index),
                    }
                    sparse_feature_refs.append(row)
                if len(sparse_feature_refs) >= max_ref_rows:
                    break
            if len(sparse_feature_refs) >= max_ref_rows:
                break
        ood_reference = _build_dual_ood_reference(
            reference_conditions=sparse_condition_refs,
            reference_features=sparse_feature_refs,
            default_threshold=float(params.get("ood_threshold", 3.0)),
        )
        runtime.payload["train_ood_reference"] = dict(ood_reference)
        ood_reference_path = write_json(stage_dirs["outputs"] / "train_ood_reference.json", ood_reference)

        feature_contract_path = write_json(
            stage_dirs["outputs"] / "feature_contract.json",
            result.feature_contract.to_dict(),
        )
        condition_scaler = dict(result.condition_scaler)
        condition_vector_diagnostics = {
            "dim": 0,
            "nonzero_count": 0,
            "min": 0.0,
            "max": 0.0,
        }
        if sparse_condition_refs:
            cond_vec = result.model.encode_conditions(sparse_condition_refs[0])
            if cond_vec:
                condition_vector_diagnostics = {
                    "dim": int(len(cond_vec)),
                    "nonzero_count": int(sum(1 for value in cond_vec if abs(float(value)) > 1e-12)),
                    "min": float(min(cond_vec)),
                    "max": float(max(cond_vec)),
                }
        model_state = {
            "model_name": "sparse_vn_student",
            "variant_id": "student",
            "model_type": type(result.model).__name__,
            "model_backend": "sparse_tensor_checkpoint",
            "checkpoint_path": str(result.checkpoint_path),
            "feature_contract_path": str(feature_contract_path),
            "feature_contract": result.feature_contract.to_dict(),
            "condition_scaler": condition_scaler,
            "ood_reference_path": str(ood_reference_path),
            "ood_reference": dict(ood_reference),
            "model_architecture": model_architecture,
            "device": cfg.device,
            "state": {},
        }
        state_path = write_json(stage_dirs["outputs"] / "model_state.json", model_state)
        metrics_path = write_json(
            stage_dirs["outputs"] / "train_metrics.json",
            {
                "best_model": "sparse_vn_student",
                "best_variant_id": "student",
                "best_metrics": dict(result.metrics),
                "comparisons": [dict(row) for row in result.comparisons],
                "metric_rows": [dict(row) for row in result.metric_rows],
                "split_info": dict(split_info),
                "seed": seed,
                "mode": "sparse_distill",
            },
        )
        comparison_path = write_csv(
            stage_dirs["outputs"] / "train_model_comparison.csv",
            [dict(row) for row in result.comparisons],
            [
                "model",
                "mae",
                "rmse",
                "target_mean",
                "prediction_mean",
                "cv_valid_mae",
                "cv_valid_rmse",
                "split",
                "status",
                "rank",
                "is_best",
                "variant_id",
                "error",
            ],
        )
        metric_rows_path = write_csv(
            stage_dirs["outputs"] / "train_metrics_long.csv",
            [dict(row) for row in result.metric_rows],
            ["model", "variant_id", "split", "metric", "value", "is_best"],
        )
        distill_path = write_json(stage_dirs["outputs"] / "distill_metrics.json", dict(result.distill_metrics))
        temporal_diag_path = write_json(
            stage_dirs["outputs"] / "train_temporal_diagnostics.json",
            dict(result.temporal_diagnostics),
        )
        distill_rows_path = write_csv(
            stage_dirs["outputs"] / "distill_metrics_long.csv",
            [
                {"model": "sparse_vn_teacher", "split": "train", "metric": "teacher_mae", "value": float(result.distill_metrics.get("teacher_mae", 0.0))},
                {"model": "sparse_vn_student", "split": "train", "metric": "student_mae", "value": float(result.distill_metrics.get("student_mae", 0.0))},
                {"model": "sparse_vn_student", "split": "train", "metric": "distill_gap", "value": float(result.distill_metrics.get("distill_gap", 0.0))},
                {"model": "sparse_vn_student", "split": "train", "metric": "vn_rmse", "value": float(result.distill_metrics.get("vn_rmse", 0.0))},
                {"model": "sparse_vn_student", "split": "train", "metric": "rollout_loss", "value": float(result.distill_metrics.get("rollout_loss", 0.0))},
                {"model": "sparse_vn_student", "split": "train", "metric": "rollout_short_window_error", "value": float(result.distill_metrics.get("rollout_loss", 0.0))},
                {"model": "sparse_vn_student", "split": "train", "metric": "best_valid_mae", "value": float(result.distill_metrics.get("best_valid_mae", result.metrics.get("cv_valid_mae", 0.0)))},
                {"model": "sparse_vn_student", "split": "cv", "metric": "valid_mae", "value": float(result.distill_metrics.get("cv_valid_mae", result.metrics.get("cv_valid_mae", 0.0)))},
                {"model": "sparse_vn_student", "split": "cv", "metric": "valid_rmse", "value": float(result.distill_metrics.get("cv_valid_rmse", result.metrics.get("cv_valid_rmse", 0.0)))},
            ],
            ["model", "split", "metric", "value"],
        )
        prediction_rows: list[dict[str, Any]] = []
        prediction_rows.extend(
            {
                "y_true": float(y_true),
                "y_pred": float(y_pred),
                "split": "train",
                "model": "sparse_vn_student",
            }
            for y_true, y_pred in zip(result.train_true, result.train_pred)
        )
        prediction_rows.extend(
            {
                "y_true": float(y_true),
                "y_pred": float(y_pred),
                "split": "cv_valid",
                "model": "sparse_vn_student",
            }
            for y_true, y_pred in zip(result.valid_true, result.valid_pred)
        )
        predictions_path = write_csv(
            stage_dirs["outputs"] / "train_predictions.csv",
            prediction_rows,
            ["y_true", "y_pred", "split", "model"],
        )
        viz_cfg = self._resolve_visualization_config(runtime=runtime, params=params, warnings=warnings)
        viz_result = render_train_output_visuals(
            output_dir=stage_dirs["outputs"],
            metric_files=[metrics_path, metric_rows_path, distill_path, distill_rows_path],
            predictions_csv=predictions_path,
            learning_curves=viz_enabled(viz_cfg, "train.learning_curves", True),
            scatter_gt_pred=viz_enabled(viz_cfg, "train.scatter_gt_pred", True),
            r2_enabled=viz_enabled(viz_cfg, "train.r2", True),
            dpi=int(viz_cfg.get("export", {}).get("dpi", 140)) if isinstance(viz_cfg.get("export"), Mapping) else 140,
        )

        metrics_out = {
            "mae": float(result.metrics.get("mae", 0.0)),
            "rmse": float(result.metrics.get("rmse", 0.0)),
            "student_mae": float(result.metrics.get("student_mae", result.metrics.get("mae", 0.0))),
            "teacher_mae": float(result.metrics.get("teacher_mae", 0.0)),
            "distill_gap": float(result.metrics.get("distill_gap", 0.0)),
            "vn_rmse": float(result.metrics.get("vn_rmse", result.metrics.get("rmse", 0.0))),
            "rollout_loss": float(result.metrics.get("rollout_loss", 0.0)),
            "rollout_short_window_error": float(result.metrics.get("rollout_loss", 0.0)),
            "best_epoch": float(result.metrics.get("best_epoch", 0.0)),
            "best_valid_mae": float(result.metrics.get("best_valid_mae", result.metrics.get("cv_valid_mae", 0.0))),
            "stopped_early": float(result.metrics.get("stopped_early", 0.0)),
            "condition_score_mean": float(ood_reference["condition"]["mean"]),
            "feature_score_mean": float(ood_reference["feature"]["mean"]),
        }
        if "cv_valid_mae" in result.metrics:
            metrics_out["cv_valid_mae"] = float(result.metrics["cv_valid_mae"])
        if "cv_valid_rmse" in result.metrics:
            metrics_out["cv_valid_rmse"] = float(result.metrics["cv_valid_rmse"])

        return StageResult(
            stage=self.name,
            status="ok",
            metrics=metrics_out,
            artifacts=[
                ArtifactRef(name="model_state", path=str(state_path), kind="json"),
                ArtifactRef(name="feature_contract", path=str(feature_contract_path), kind="json"),
                ArtifactRef(name="train_ood_reference", path=str(ood_reference_path), kind="json"),
                ArtifactRef(name="sparse_checkpoint", path=str(result.checkpoint_path), kind="pt"),
                ArtifactRef(name="train_metrics", path=str(metrics_path), kind="json"),
                ArtifactRef(name="train_model_comparison", path=str(comparison_path), kind="csv"),
                ArtifactRef(name="train_metrics_long", path=str(metric_rows_path), kind="csv"),
                ArtifactRef(name="distill_metrics", path=str(distill_path), kind="json"),
                ArtifactRef(name="train_temporal_diagnostics", path=str(temporal_diag_path), kind="json"),
                ArtifactRef(name="distill_metrics_long", path=str(distill_rows_path), kind="csv"),
                ArtifactRef(name="train_predictions", path=str(predictions_path), kind="csv"),
                ArtifactRef(name="train_visualization_manifest", path=str(viz_result["manifest_path"]), kind="json"),
            ],
            details={
                "mode": "sparse_distill",
                "best_model": "sparse_vn_student",
                "best_variant_id": "student",
                "num_model_variants": 2,
                "num_successful_variants": 2,
                "split_info": dict(split_info),
                "seed": seed,
                "distill_weights": {
                    "alpha": float(cfg.alpha),
                    "beta": float(cfg.beta),
                    "gamma": float(cfg.gamma),
                },
                "training_controls": {
                    "early_stopping_patience": int(cfg.early_stopping_patience),
                    "lr_scheduler": str(cfg.lr_scheduler),
                    "grad_clip_norm": float(cfg.grad_clip_norm),
                    "rollout_loss_enabled": bool(cfg.rollout_loss_enabled),
                    "rollout_k": int(cfg.rollout_k),
                    "rollout_weight": float(cfg.rollout_weight),
                    "temporal_step_weight": str(cfg.temporal_step_weight),
                    "step_sampling_policy": str(cfg.step_sampling_policy),
                },
                "model_architecture": model_architecture,
                "condition_vector_diagnostics": condition_vector_diagnostics,
                "train_temporal_diagnostics_path": str(temporal_diag_path),
                "ood_reference_path": str(ood_reference_path),
                "visualization_manifest_path": str(viz_result["manifest_path"]),
                "input_refs": input_refs,
                "warnings": warnings,
                "sparse_backend": detect_runtime_capabilities().missing_summary(),
                "fallback_reason": "",
            },
        )

    def _run_tabular(self, runtime: Any, stage_dirs: dict[str, Path], params: dict[str, Any], external_inputs: Mapping[str, str], input_refs: dict[str, str], warnings: list[str], forced_model_name: str | None = None, mode_label: str = "tabular", fallback_reason: str = "") -> StageResult:
        features = runtime.payload.get("processed_features") or runtime.payload.get("features")
        targets = runtime.payload.get("processed_targets") or runtime.payload.get("targets")
        preprocess_bundle = runtime.payload.get("preprocess_bundle")

        if not isinstance(features, list) or not features:
            source_path = external_inputs.get("processed_features_csv") or external_inputs.get("features_csv")
            if source_path:
                resolved = Path(source_path)
                if not resolved.exists() or not resolved.is_file():
                    raise ValueError(f"train processed_features_csv does not exist: {resolved}")
                features = self._read_feature_rows(resolved)
                input_refs["processed_features_csv"] = str(resolved)

        if not isinstance(targets, list) or not targets:
            source_path = external_inputs.get("processed_targets_csv") or external_inputs.get("targets_csv")
            if source_path:
                resolved = Path(source_path)
                if not resolved.exists() or not resolved.is_file():
                    raise ValueError(f"train processed_targets_csv does not exist: {resolved}")
                targets = self._read_target_values(resolved)
                input_refs["processed_targets_csv"] = str(resolved)

        if not isinstance(preprocess_bundle, Mapping):
            source_path = external_inputs.get("preprocess_bundle_json")
            if source_path:
                resolved = Path(source_path)
                if not resolved.exists() or not resolved.is_file():
                    raise ValueError(f"train preprocess_bundle_json does not exist: {resolved}")
                preprocess_bundle = self._read_json_mapping(resolved)
                runtime.payload["preprocess_bundle"] = preprocess_bundle
                input_refs["preprocess_bundle_json"] = str(resolved)

        if not isinstance(features, list) or not features:
            raise ValueError("train stage requires processed_features/features")
        if not isinstance(targets, list) or not targets:
            raise ValueError("train stage requires processed_targets/targets")

        model_kwargs_raw = params.get("model_kwargs", {})
        model_kwargs = model_kwargs_raw if isinstance(model_kwargs_raw, dict) else {}
        model_variants = self._resolve_model_variants(
            params=params,
            default_model_kwargs=model_kwargs,
            forced_model_name=forced_model_name,
        )

        targets_float = [float(value) for value in targets]
        if len(features) != len(targets_float):
            raise ValueError("train stage requires features and targets with the same number of rows")

        tabular_contract = self._build_tabular_feature_contract(features=features, params=params)

        row_refs: list[dict[str, Any]] = []
        if isinstance(preprocess_bundle, Mapping):
            candidate = preprocess_bundle.get("row_refs")
            if isinstance(candidate, list):
                row_refs = [dict(row) for row in candidate if isinstance(row, Mapping)]

        seed = int(params.get("seed", 0))
        cv_folds = int(params.get("cv_folds", 3))
        train_folds, valid_folds, split_info = self._build_cv_folds(
            row_refs=row_refs,
            n_samples=len(features),
            cv_folds=cv_folds,
            seed=seed,
        )
        split_info = self._normalize_split_info(split_info, loader_mode_default="in_memory")
        strict_split = bool(params.get("strict_split", False))
        if strict_split and int(split_info.get("num_valid_runs", 0)) < 1:
            reason = str(split_info.get("reason", "insufficient_unique_runs")) or "insufficient_unique_runs"
            raise ValueError(f"strict_split=true requires num_valid_runs>0 ({reason})")
        split_info["strict_split_enforced"] = bool(strict_split and int(split_info.get("num_valid_runs", 0)) > 0)

        comparisons: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        successful_rows: list[dict[str, Any]] = []
        best_model = None
        best_metrics = None
        best_name = ""
        best_variant_id = ""
        seen_warnings = {str(item) for item in warnings}

        for variant in model_variants:
            model_name = str(variant["model"])
            resolved_model_name, alias_warning = resolve_model_alias(model_name)
            if alias_warning and alias_warning not in seen_warnings:
                warnings.append(alias_warning)
                seen_warnings.add(alias_warning)
            variant_id = str(variant["variant_id"])
            variant_kwargs = variant["model_kwargs"] if isinstance(variant.get("model_kwargs"), dict) else {}
            try:
                cv_summary, fold_rows = self._cross_validate(
                    model_name=resolved_model_name,
                    model_kwargs=variant_kwargs,
                    features=features,
                    targets=targets_float,
                    train_folds=train_folds,
                    valid_folds=valid_folds,
                )
                model, train_metrics = self._train_one(
                    model_name=resolved_model_name,
                    model_kwargs=variant_kwargs,
                    features=features,
                    targets=targets_float,
                )
                row: dict[str, Any] = {
                    "variant_id": variant_id,
                    "model": resolved_model_name,
                    "split": "train",
                    "status": "ok",
                    "error": "",
                    **{key: float(value) for key, value in train_metrics.items()},
                }
                row.update(cv_summary)
                comparisons.append(row)
                successful_rows.append(row)

                for metric_name in sorted(_metric_registry().keys()):
                    metric_rows.append(
                        {
                            "variant_id": variant_id,
                            "model": resolved_model_name,
                            "split": "train",
                            "metric": metric_name,
                            "value": float(train_metrics[metric_name]),
                            "is_best": 0,
                        }
                    )
                if cv_summary:
                    for fold in fold_rows:
                        metric_rows.append(
                            {
                                "variant_id": variant_id,
                                "model": resolved_model_name,
                                "split": "cv",
                                "metric": "valid_mae",
                                "value": float(fold["valid_mae"]),
                                "is_best": 0,
                            }
                        )

                compare_value = float(cv_summary.get("cv_valid_mae", train_metrics["mae"]))
                current_best = float(best_metrics.get("compare_value", 1e99)) if isinstance(best_metrics, dict) else 1e99
                if best_metrics is None or compare_value < current_best:
                    best_model = model
                    best_metrics = {**train_metrics, "compare_value": compare_value, **cv_summary}
                    best_name = resolved_model_name
                    best_variant_id = variant_id
            except Exception as exc:
                comparisons.append(
                    {
                        "variant_id": variant_id,
                        "model": resolved_model_name,
                        "split": "train",
                        "status": "error",
                        "error": str(exc),
                        "mae": None,
                        "rmse": None,
                        "target_mean": None,
                        "prediction_mean": None,
                        "cv_valid_mae": None,
                        "cv_valid_rmse": None,
                    }
                )

        if best_model is None or best_metrics is None:
            raise ValueError("train stage failed: no model variant completed successfully")

        ranked_rows = sorted(
            [row for row in successful_rows if row.get("status") == "ok"],
            key=lambda row: float(row.get("cv_valid_mae", row["mae"])),
        )
        for rank, row in enumerate(ranked_rows, start=1):
            row["rank"] = rank
            row["is_best"] = 1 if str(row["variant_id"]) == best_variant_id else 0
        for row in comparisons:
            row.setdefault("rank", 0)
            row.setdefault("is_best", 0)

        for row in metric_rows:
            if str(row.get("variant_id")) == best_variant_id:
                row["is_best"] = 1

        if split_info.get("mode") == "disabled":
            warnings.append("cv disabled due to insufficient run-level split information")

        runtime.payload["trained_model"] = best_model
        runtime.payload["trained_model_name"] = best_name
        runtime.payload["train_metrics"] = best_metrics
        runtime.payload["train_comparisons"] = comparisons
        runtime.payload["train_metric_rows"] = metric_rows
        runtime.payload["train_split_info"] = split_info
        runtime.payload["feature_contract"] = dict(tabular_contract)

        condition_refs = _condition_rows_from_features(features)
        feature_refs = [{str(k): float(v) for k, v in row.items()} for row in features[:200]]
        ood_reference = _build_dual_ood_reference(
            reference_conditions=condition_refs,
            reference_features=feature_refs,
            default_threshold=float(params.get("ood_threshold", 3.0)),
        )
        runtime.payload["train_ood_reference"] = dict(ood_reference)
        ood_reference_path = write_json(stage_dirs["outputs"] / "train_ood_reference.json", ood_reference)
        feature_contract_path = write_json(
            stage_dirs["outputs"] / "feature_contract.json",
            dict(tabular_contract),
        )

        model_state = {
            "model_name": best_name,
            "variant_id": best_variant_id,
            "model_type": type(best_model).__name__,
            "model_backend": "tabular",
            "checkpoint_path": "",
            "feature_contract_path": str(feature_contract_path),
            "feature_contract": dict(tabular_contract),
            "ood_reference_path": str(ood_reference_path),
            "ood_reference": dict(ood_reference),
            "state": {
                key: value
                for key, value in getattr(best_model, "__dict__", {}).items()
                if isinstance(value, (int, float, str, bool, list, dict, type(None)))
            },
        }
        state_path = write_json(stage_dirs["outputs"] / "model_state.json", model_state)
        metrics_path = write_json(
            stage_dirs["outputs"] / "train_metrics.json",
            {
                "best_model": best_name,
                "best_variant_id": best_variant_id,
                "best_metrics": {key: float(value) for key, value in best_metrics.items() if isinstance(value, (int, float))},
                "comparisons": comparisons,
                "metric_rows": metric_rows,
                "split_info": split_info,
                "seed": seed,
                "mode": mode_label,
            },
        )
        comparison_path = write_csv(
            stage_dirs["outputs"] / "train_model_comparison.csv",
            [dict(row) for row in comparisons],
            [
                "model",
                "mae",
                "rmse",
                "target_mean",
                "prediction_mean",
                "cv_valid_mae",
                "cv_valid_rmse",
                "split",
                "status",
                "rank",
                "is_best",
                "variant_id",
                "error",
            ],
        )
        metric_rows_path = write_csv(
            stage_dirs["outputs"] / "train_metrics_long.csv",
            [dict(row) for row in metric_rows],
            ["model", "variant_id", "split", "metric", "value", "is_best"],
        )
        train_predictions_rows = [
            {
                "y_true": float(target_value),
                "y_pred": float(best_model.predict(feature_row)),
                "split": "train",
                "model": str(best_name),
            }
            for feature_row, target_value in zip(features, targets_float)
        ]
        predictions_path = write_csv(
            stage_dirs["outputs"] / "train_predictions.csv",
            train_predictions_rows,
            ["y_true", "y_pred", "split", "model"],
        )
        viz_cfg = self._resolve_visualization_config(runtime=runtime, params=params, warnings=warnings)
        viz_result = render_train_output_visuals(
            output_dir=stage_dirs["outputs"],
            metric_files=[metrics_path, comparison_path, metric_rows_path],
            predictions_csv=predictions_path,
            learning_curves=viz_enabled(viz_cfg, "train.learning_curves", True),
            scatter_gt_pred=viz_enabled(viz_cfg, "train.scatter_gt_pred", True),
            r2_enabled=viz_enabled(viz_cfg, "train.r2", True),
            dpi=int(viz_cfg.get("export", {}).get("dpi", 140)) if isinstance(viz_cfg.get("export"), Mapping) else 140,
        )

        result_metrics = {
            "mae": float(best_metrics["mae"]),
            "rmse": float(best_metrics["rmse"]),
            "target_mean": float(best_metrics["target_mean"]),
            "prediction_mean": float(best_metrics["prediction_mean"]),
            "condition_score_mean": float(ood_reference["condition"]["mean"]),
            "feature_score_mean": float(ood_reference["feature"]["mean"]),
            "rollout_short_window_error": float(best_metrics.get("cv_valid_rmse", best_metrics["rmse"])),
        }
        if "cv_valid_mae" in best_metrics:
            result_metrics["cv_valid_mae"] = float(best_metrics["cv_valid_mae"])
        if "cv_valid_rmse" in best_metrics:
            result_metrics["cv_valid_rmse"] = float(best_metrics["cv_valid_rmse"])

        return StageResult(
            stage=self.name,
            status="ok",
            metrics=result_metrics,
            artifacts=[
                ArtifactRef(name="model_state", path=str(state_path), kind="json"),
                ArtifactRef(name="feature_contract", path=str(feature_contract_path), kind="json"),
                ArtifactRef(name="train_ood_reference", path=str(ood_reference_path), kind="json"),
                ArtifactRef(name="train_metrics", path=str(metrics_path), kind="json"),
                ArtifactRef(name="train_model_comparison", path=str(comparison_path), kind="csv"),
                ArtifactRef(name="train_metrics_long", path=str(metric_rows_path), kind="csv"),
                ArtifactRef(name="train_predictions", path=str(predictions_path), kind="csv"),
                ArtifactRef(name="train_visualization_manifest", path=str(viz_result["manifest_path"]), kind="json"),
            ],
            details={
                "mode": mode_label,
                "best_model": best_name,
                "best_variant_id": best_variant_id,
                "num_model_variants": len(model_variants),
                "num_successful_variants": len(successful_rows),
                "split_info": split_info,
                "seed": seed,
                "ood_reference_path": str(ood_reference_path),
                "visualization_manifest_path": str(viz_result["manifest_path"]),
                "input_refs": input_refs,
                "warnings": warnings,
                "fallback_reason": str(fallback_reason),
            },
        )

    def run(self, runtime: Any, stage_dirs: dict[str, Path]) -> StageResult:
        params = runtime.stage_params("train")
        external_inputs = self._stage_external_inputs(runtime)
        input_refs: dict[str, str] = {}
        warnings: list[str] = []

        mode = str(params.get("mode", "tabular")).strip().lower() or "tabular"
        backend: TrainBackend
        if mode == "sparse_distill":
            backend = SparseDistillTrainBackend()
        else:
            backend = TabularTrainBackend()
        return backend.run(
            self,
            runtime=runtime,
            stage_dirs=stage_dirs,
            params=params,
            external_inputs=external_inputs,
            input_refs=input_refs,
            warnings=warnings,
        )
