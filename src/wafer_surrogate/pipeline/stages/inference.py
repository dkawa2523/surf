from __future__ import annotations

import csv
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from wafer_surrogate.core import (
    assess_dual_ood,
    assert_feature_contract_compatible,
    frame_mean,
    load_feature_contract,
    normalize_feature_contract,
    rollout,
    to_float_map,
    validate_predict_vn_contract,
    validate_rows_against_feature_contract,
    write_csv,
    write_json,
)
from wafer_surrogate.data.sem import SemFeatureError, load_sem_features
from wafer_surrogate.data.synthetic import SyntheticSDFDataset, SyntheticSDFRun
from wafer_surrogate.inference.calibrate import (
    OptionalDependencyUnavailable,
    calibrate_latent_map_with_observation,
    sample_latent_posterior_sbi,
    train_latent_posterior_sbi,
)
from wafer_surrogate.models import make_model
from wafer_surrogate.models.sparse_unet_film import OptionalSparseDependencyUnavailable, SparseTensorVnModel
from wafer_surrogate.metrics import compute_temporal_diagnostics
from wafer_surrogate.observation import make_observation_model
from wafer_surrogate.optimization import run_optimization_engine
from wafer_surrogate.pipeline.types import ArtifactRef, StageResult
from wafer_surrogate.prior import make_shape_prior
from wafer_surrogate.runtime import detect_runtime_capabilities
from wafer_surrogate.viz.utils import load_pyplot, resolve_visualization_config, viz_enabled, write_visualization_manifest


class InferenceBackend(Protocol):
    mode: str


@dataclass(frozen=True)
class SingleInferenceBackend:
    mode: str = "single"


@dataclass(frozen=True)
class BatchInferenceBackend:
    mode: str = "batch"


@dataclass(frozen=True)
class OptimizeInferenceBackend:
    mode: str = "optimize"


def _resolve_inference_backend(mode: str) -> InferenceBackend:
    normalized = str(mode).strip().lower()
    if normalized == "single":
        return SingleInferenceBackend()
    if normalized == "batch":
        return BatchInferenceBackend()
    if normalized == "optimize":
        return OptimizeInferenceBackend()
    raise ValueError(f"unsupported inference mode: {mode}")


def _to_float_nested(frame: Any) -> Any:
    if hasattr(frame, "tolist"):
        frame = frame.tolist()
    if isinstance(frame, Sequence) and not isinstance(frame, (str, bytes, bytearray)):
        return [_to_float_nested(cell) for cell in frame]
    return float(frame)


def _dataset_from_json(path: Path) -> SyntheticSDFDataset:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, Mapping):
        raise ValueError(f"inference dataset_json must be mapping: {path}")
    runs_payload = payload.get("runs")
    if not isinstance(runs_payload, list) or not runs_payload:
        raise ValueError(f"inference dataset_json requires non-empty runs: {path}")
    runs: list[SyntheticSDFRun] = []
    for index, run in enumerate(runs_payload):
        if not isinstance(run, Mapping):
            raise ValueError(f"inference runs[{index}] must be mapping")
        recipe_raw = run.get("recipe", {})
        if not isinstance(recipe_raw, Mapping):
            raise ValueError(f"inference runs[{index}].recipe must be mapping")
        phi_t_raw = run.get("phi_t")
        if not isinstance(phi_t_raw, list) or not phi_t_raw:
            raise ValueError(f"inference runs[{index}].phi_t must be non-empty list")
        phi_t = []
        for frame in phi_t_raw:
            if hasattr(frame, "tolist"):
                frame = frame.tolist()
            phi_t.append([[float(cell) for cell in row] for row in frame])
        runs.append(
            SyntheticSDFRun(
                run_id=str(run.get("run_id", f"run_{index:03d}")),
                dt=float(run.get("dt", 0.1)),
                recipe={str(k): float(v) for k, v in recipe_raw.items()},
                phi_t=phi_t,
            )
        )
    return SyntheticSDFDataset(runs=runs)


def _load_feature_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = [dict(row) for row in csv.DictReader(fp)]
    return [{str(key): float(value) for key, value in row.items()} for row in rows]


def _load_condition_rows(path: str | None) -> list[dict[str, float]]:
    if path is None:
        return []
    csv_path = Path(path)
    if not csv_path.exists() or not csv_path.is_file():
        raise ValueError(f"conditions_csv does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        rows = [dict(row) for row in csv.DictReader(fp)]
    out: list[dict[str, float]] = []
    for row in rows:
        out.append({str(key): float(value) for key, value in row.items()})
    return out


def _load_model_from_state(path: Path) -> tuple[Any, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, Mapping):
        raise ValueError(f"model_state_json must be mapping: {path}")
    backend = str(payload.get("model_backend", "")).strip().lower()
    if backend == "sparse_tensor_checkpoint":
        checkpoint_path = payload.get("checkpoint_path")
        if checkpoint_path is None:
            raise ValueError("model_state sparse backend requires checkpoint_path")
        checkpoint = Path(str(checkpoint_path))
        if not checkpoint.is_absolute() and not checkpoint.exists():
            checkpoint = (path.parent / checkpoint).resolve()
        if not checkpoint.exists() or not checkpoint.is_file():
            raise ValueError(f"sparse checkpoint does not exist: {checkpoint}")
        try:
            model = SparseTensorVnModel.from_checkpoint(checkpoint, device=str(payload.get("device", "cpu")))
            scaler = payload.get("condition_scaler")
            if isinstance(scaler, Mapping):
                model.condition_scaler = {str(k): v for k, v in scaler.items()}
            return model, {str(k): v for k, v in payload.items()}
        except OptionalSparseDependencyUnavailable as exc:
            raise ValueError(f"sparse checkpoint load failed due to missing optional dependency: {exc}") from exc

    model_name = str(payload.get("model_name", "baseline_vn_linear_trainable"))
    model = make_model(model_name)
    state = payload.get("state")
    if isinstance(state, Mapping):
        for key, value in state.items():
            setattr(model, str(key), value)
    return model, {str(k): v for k, v in payload.items()}


def _condition_feature_projection(base_row: Mapping[str, float], conditions: Mapping[str, float]) -> dict[str, float]:
    out = {str(k): float(v) for k, v in base_row.items()}
    for key, value in conditions.items():
        out[str(key)] = float(value)
        feat_key = f"feat_{key}"
        if feat_key in out:
            out[feat_key] = float(value)
        cond_key = f"cond_{key}"
        if cond_key in out:
            out[cond_key] = float(value)
    return out


def _extract_condition_reference(dataset: SyntheticSDFDataset | None, feature_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    if dataset is not None and dataset.runs:
        return [to_float_map(run.recipe) for run in dataset.runs]

    refs: list[dict[str, float]] = []
    for row in feature_rows:
        cond: dict[str, float] = {}
        for key, value in row.items():
            if key.startswith("feat_"):
                cond[key[5:]] = float(value)
        if cond:
            refs.append(cond)
    return refs


def _extract_feature_reference(feature_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    return [{str(k): float(v) for k, v in row.items()} for row in feature_rows]


def _select_template_run(dataset: SyntheticSDFDataset, template_run_id: str) -> SyntheticSDFRun:
    if not dataset.runs:
        raise ValueError("inference dataset contains no runs")
    requested = str(template_run_id).strip()
    if requested:
        for run in dataset.runs:
            if str(run.run_id) == requested:
                return run
        raise ValueError(f"inference template_run_id not found in dataset: {requested}")
    return sorted(dataset.runs, key=lambda run: str(run.run_id))[0]


def _run_rollout(
    template: SyntheticSDFRun,
    model: Any,
    conditions: Mapping[str, float],
    *,
    simulation_options: Mapping[str, object] | None,
) -> list[Any]:
    run = SyntheticSDFRun(
        run_id=template.run_id,
        dt=float(template.dt),
        recipe={str(k): float(v) for k, v in conditions.items()},
        phi_t=template.phi_t,
    )
    return rollout(run, model, simulation_options=simulation_options)


def _ood_status_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("ood_status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    qq = max(0.0, min(1.0, float(q)))
    pos = qq * float(len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - float(lo)
    return float((1.0 - frac) * ordered[lo] + frac * ordered[hi])


def _ood_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    condition_scores = [
        float(value)
        for value in (row.get("condition_score") for row in rows)
        if isinstance(value, (int, float))
    ]
    feature_scores = [
        float(value)
        for value in (row.get("feature_score") for row in rows)
        if isinstance(value, (int, float))
    ]
    total = max(1, len(rows))
    out_of_domain = sum(1 for row in rows if str(row.get("ood_status", "")) == "out_of_domain")
    return {
        "condition_score_mean": (float(fmean(condition_scores)) if condition_scores else None),
        "condition_score_p95": _percentile(condition_scores, 0.95),
        "condition_score_max": (float(max(condition_scores)) if condition_scores else None),
        "feature_score_mean": (float(fmean(feature_scores)) if feature_scores else None),
        "feature_score_p95": _percentile(feature_scores, 0.95),
        "feature_score_max": (float(max(feature_scores)) if feature_scores else None),
        "num_out_of_domain": int(out_of_domain),
        "out_of_domain_ratio": float(out_of_domain) / float(total),
    }


def _estimate_feature_importance(history: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    rows = [row for row in history if isinstance(row.get("conditions"), Mapping)]
    if not rows:
        return {}
    keys = sorted({str(k) for row in rows for k in row["conditions"].keys()})  # type: ignore[index]
    if not keys:
        return {}
    objectives = [float(row.get("objective", 0.0)) for row in rows]
    n = len(objectives)
    if n < 2:
        return {key: 0.0 for key in keys}
    mean_y = fmean(objectives)
    var_y = sum((yy - mean_y) ** 2 for yy in objectives)
    if abs(var_y) <= 1e-12:
        return {key: 0.0 for key in keys}

    scores: dict[str, float] = {}
    for key in keys:
        xs = [float(row["conditions"].get(key, 0.0)) for row in rows]  # type: ignore[index]
        mean_x = fmean(xs)
        var_x = sum((xx - mean_x) ** 2 for xx in xs)
        if abs(var_x) <= 1e-12:
            scores[key] = 0.0
            continue
        cov = sum((xx - mean_x) * (yy - mean_y) for xx, yy in zip(xs, objectives))
        corr = cov / ((var_x * var_y) ** 0.5)
        scores[key] = abs(float(corr))
    return scores


def _write_ood_report(
    *,
    stage_dirs: Mapping[str, Path],
    mode: str,
    rows: Sequence[Mapping[str, object]],
    threshold: float,
    extras: Mapping[str, object] | None = None,
) -> Path:
    payload: dict[str, object] = {
        "mode": mode,
        "ood_threshold": float(threshold),
        "num_records": len(rows),
        "status_counts": _ood_status_counts(rows),
        "summary": _ood_summary(rows),
        "records": [dict(row) for row in rows],
    }
    if extras:
        payload.update({str(key): value for key, value in extras.items()})
    return write_json(stage_dirs["outputs"] / f"inference_{mode}_ood_report.json", payload)


def _write_temporal_diagnostics(
    *,
    stage_dirs: Mapping[str, Path],
    mode: str,
    rows: Sequence[Mapping[str, Any]],
    extras: Mapping[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "mode": mode,
        "num_records": len(rows),
        "records": [dict(row) for row in rows],
    }
    if extras:
        payload.update({str(key): value for key, value in extras.items()})
    return write_json(stage_dirs["outputs"] / "temporal_diagnostics.json", payload)


def _plot_temporal_diagnostics_rows(
    *,
    stage_dirs: Mapping[str, Path],
    mode: str,
    rows: Sequence[Mapping[str, Any]],
    enabled: bool,
    dpi: int,
    warnings: list[str],
) -> Path | None:
    if not enabled:
        warnings.append("inference temporal diagnostics plot skipped: disabled by visualization config")
        return None
    plt = load_pyplot()
    if plt is None:
        warnings.append("inference temporal diagnostics plot skipped: matplotlib unavailable")
        return None
    if not rows:
        warnings.append("inference temporal diagnostics plot skipped: no temporal rows")
        return None

    indices = list(range(len(rows)))
    keys = (
        "delta_phi_sign_agreement",
        "early_window_error",
        "late_window_error",
        "r2_all_frames",
        "r2_final_frame",
    )
    series: dict[str, list[float]] = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        if any(isinstance(value, (int, float)) for value in values):
            series[key] = [float(value) if isinstance(value, (int, float)) else float("nan") for value in values]
    if not series:
        warnings.append("inference temporal diagnostics plot skipped: temporal metrics missing")
        return None

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.8))
    top = axes[0]
    bottom = axes[1]
    for key in ("delta_phi_sign_agreement", "r2_all_frames", "r2_final_frame"):
        values = series.get(key)
        if values:
            top.plot(indices, values, marker="o", linewidth=1.4, label=key)
    for key in ("early_window_error", "late_window_error"):
        values = series.get(key)
        if values:
            bottom.plot(indices, values, marker="o", linewidth=1.4, label=key)

    top.set_title(f"Temporal Diagnostics ({mode})")
    top.set_ylabel("score")
    top.grid(alpha=0.25)
    if top.lines:
        top.legend(loc="best", fontsize=8)

    bottom.set_xlabel("record index")
    bottom.set_ylabel("error")
    bottom.grid(alpha=0.25)
    if bottom.lines:
        bottom.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_path = stage_dirs["outputs"] / f"temporal_diagnostics_{mode}.png"
    fig.savefig(out_path, dpi=max(72, int(dpi)))
    plt.close(fig)
    return out_path


def _load_json_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, Mapping):
        raise ValueError(f"json mapping expected: {path}")
    return {str(k): v for k, v in payload.items()}


def _inject_latent(conditions: Mapping[str, float], z: Sequence[float]) -> dict[str, float]:
    out = {str(k): float(v) for k, v in conditions.items()}
    for idx, value in enumerate(z):
        out[f"z_{idx:02d}"] = float(value)
    return out


def _condition_vector_diagnostics(model: Any, conditions: Mapping[str, float]) -> dict[str, Any] | None:
    encode = getattr(model, "encode_conditions", None)
    if encode is None:
        return None
    try:
        cond_vec = [float(v) for v in encode({str(k): float(v) for k, v in conditions.items()})]
    except Exception:
        return None
    if not cond_vec:
        return {"dim": 0, "nonzero_count": 0, "min": 0.0, "max": 0.0}
    return {
        "dim": int(len(cond_vec)),
        "nonzero_count": int(sum(1 for value in cond_vec if abs(float(value)) > 1e-12)),
        "min": float(min(cond_vec)),
        "max": float(max(cond_vec)),
    }


def _load_latent_prior_payload(path: Path, *, latent_dim: int) -> dict[str, list[float]]:
    payload = _load_json_mapping(path)
    out: dict[str, list[float]] = {
        "z_init": [0.0 for _ in range(latent_dim)],
        "prior_mean": [0.0 for _ in range(latent_dim)],
        "prior_std": [1.0 for _ in range(latent_dim)],
    }
    for key in ("z_init", "prior_mean", "prior_std"):
        raw = payload.get(key)
        if raw is None:
            continue
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise ValueError(f"latent_prior_json field '{key}' must be a numeric sequence")
        values = [float(v) for v in raw][:latent_dim]
        while len(values) < latent_dim:
            values.append(0.0 if key != "prior_std" else 1.0)
        if key == "prior_std":
            values = [max(1e-6, float(v)) for v in values]
        out[key] = values
    return out


class InferenceStage:
    name = "inference"

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

    def run(self, runtime: Any, stage_dirs: dict[str, Path]) -> StageResult:
        params = runtime.stage_params("inference")
        mode = str(params.get("mode", "single")).strip().lower()
        backend = _resolve_inference_backend(mode)
        mode = backend.mode
        threshold = float(params.get("ood_threshold", 3.0))

        reinit_enabled = bool(params.get("reinit_enabled", False))
        reinit_every_n = max(1, int(params.get("reinit_every_n", 5)))
        reinit_iters = max(1, int(params.get("reinit_iters", 8)))
        reinit_dt = float(params.get("reinit_dt", 0.3))

        external_inputs = self._stage_external_inputs(runtime)
        input_refs: dict[str, str] = {}
        warnings: list[str] = []
        reinit_events: list[dict[str, Any]] = []
        artifacts: list[ArtifactRef] = []
        requested_template_run_id = str(params.get("template_run_id", "")).strip()
        runtime_config = runtime.payload.get("config")
        run_viz_cfg = runtime_config.get("visualization") if isinstance(runtime_config, Mapping) else {}
        stage_viz_cfg = params.get("visualization") if isinstance(params, Mapping) else {}
        viz_cfg = resolve_visualization_config(
            run_config=run_viz_cfg if isinstance(run_viz_cfg, Mapping) else {},
            stage_config=stage_viz_cfg if isinstance(stage_viz_cfg, Mapping) else {},
            warnings=warnings,
        )

        def _emit_viz_manifest(mode_name: str, outputs: Mapping[str, Any]) -> Path:
            manifest_payload = {
                "mode": mode_name,
                "config": viz_cfg,
                "outputs": {str(k): v for k, v in outputs.items()},
                "warnings": [str(message) for message in warnings if str(message).startswith("inference temporal diagnostics plot") or "viz config" in str(message)],
            }
            return write_visualization_manifest(
                stage_dirs["outputs"] / "visualization_manifest.json",
                manifest_payload,
            )

        model = runtime.payload.get("trained_model")
        model_state_payload: dict[str, Any] = {}
        if model is None:
            model_path = external_inputs.get("model_state_json")
            if model_path:
                resolved = Path(model_path)
                if not resolved.exists() or not resolved.is_file():
                    raise ValueError(f"inference model_state_json does not exist: {resolved}")
                model, model_state_payload = _load_model_from_state(resolved)
                input_refs["model_state_json"] = str(resolved)
        if model is None:
            raise ValueError("inference stage requires trained_model or external_inputs.model_state_json")
        validate_predict_vn_contract(model)

        expected_contract: dict[str, Any] | None = None
        runtime_contract = runtime.payload.get("feature_contract")
        if isinstance(runtime_contract, Mapping):
            expected_contract = normalize_feature_contract(
                runtime_contract,
                source="runtime.feature_contract",
                require_extended=True,
            )
        external_contract_path = external_inputs.get("feature_contract_json")
        if external_contract_path:
            external_contract = load_feature_contract(external_contract_path)
            if expected_contract is not None:
                assert_feature_contract_compatible(
                    expected=expected_contract,
                    actual=external_contract,
                    context="inference.feature_contract",
                )
            expected_contract = external_contract
            input_refs["feature_contract_json"] = str(Path(external_contract_path))

        inline_contract = model_state_payload.get("feature_contract")
        if isinstance(inline_contract, Mapping):
            normalized_inline = normalize_feature_contract(
                inline_contract,
                source="model_state.feature_contract",
                require_extended=True,
            )
            if expected_contract is not None:
                assert_feature_contract_compatible(
                    expected=expected_contract,
                    actual=normalized_inline,
                    context="inference.feature_contract",
                )
            expected_contract = normalized_inline
        feature_contract_path = model_state_payload.get("feature_contract_path")
        if isinstance(feature_contract_path, str) and feature_contract_path.strip():
            state_path_raw = input_refs.get("model_state_json")
            candidate = Path(feature_contract_path)
            if state_path_raw and not candidate.is_absolute() and not candidate.exists():
                candidate = (Path(state_path_raw).parent / candidate).resolve()
            loaded = load_feature_contract(candidate)
            if expected_contract is not None:
                assert_feature_contract_compatible(
                    expected=expected_contract,
                    actual=loaded,
                    context="inference.feature_contract",
                )
            expected_contract = loaded

        ood_reference_profile: dict[str, Any] | None = None
        runtime_ood_reference = runtime.payload.get("train_ood_reference")
        if isinstance(runtime_ood_reference, Mapping):
            ood_reference_profile = {str(k): v for k, v in runtime_ood_reference.items()}
        ood_reference_path = external_inputs.get("train_ood_reference_json")
        if ood_reference_path:
            resolved = Path(str(ood_reference_path))
            if not resolved.exists() or not resolved.is_file():
                raise ValueError(f"inference train_ood_reference_json does not exist: {resolved}")
            loaded = _load_json_mapping(resolved)
            ood_reference_profile = loaded
            input_refs["train_ood_reference_json"] = str(resolved)
        inline_ood_reference = model_state_payload.get("ood_reference")
        if isinstance(inline_ood_reference, Mapping):
            ood_reference_profile = {str(k): v for k, v in inline_ood_reference.items()}
        model_ood_reference_path = model_state_payload.get("ood_reference_path")
        if isinstance(model_ood_reference_path, str) and model_ood_reference_path.strip():
            state_path_raw = input_refs.get("model_state_json")
            candidate = Path(model_ood_reference_path)
            if state_path_raw and not candidate.is_absolute() and not candidate.exists():
                candidate = (Path(state_path_raw).parent / candidate).resolve()
            if candidate.exists() and candidate.is_file():
                ood_reference_profile = _load_json_mapping(candidate)
                input_refs["model_ood_reference_path"] = str(candidate)

        dataset = runtime.payload.get("dataset_clean") or runtime.payload.get("dataset_raw")
        if not isinstance(dataset, SyntheticSDFDataset):
            dataset_path = external_inputs.get("dataset_json")
            if dataset_path:
                resolved = Path(dataset_path)
                if not resolved.exists() or not resolved.is_file():
                    raise ValueError(f"inference dataset_json does not exist: {resolved}")
                dataset = _dataset_from_json(resolved)
                input_refs["dataset_json"] = str(resolved)

        processed_features = runtime.payload.get("processed_features")
        if not isinstance(processed_features, list):
            feature_path = external_inputs.get("processed_features_csv")
            if feature_path:
                resolved = Path(feature_path)
                if not resolved.exists() or not resolved.is_file():
                    raise ValueError(f"inference processed_features_csv does not exist: {resolved}")
                processed_features = _load_feature_rows(resolved)
                input_refs["processed_features_csv"] = str(resolved)
        feature_rows = [dict(row) for row in processed_features] if isinstance(processed_features, list) else []

        preprocess_bundle = runtime.payload.get("preprocess_bundle")
        if not isinstance(preprocess_bundle, Mapping):
            preprocess_bundle_path = external_inputs.get("preprocess_bundle_json")
            if preprocess_bundle_path:
                resolved = Path(preprocess_bundle_path)
                if not resolved.exists() or not resolved.is_file():
                    raise ValueError(f"inference preprocess_bundle_json does not exist: {resolved}")
                preprocess_bundle = _load_json_mapping(resolved)
                runtime.payload["preprocess_bundle"] = dict(preprocess_bundle)
                input_refs["preprocess_bundle_json"] = str(resolved)

        reconstruction_bundle = runtime.payload.get("reconstruction_bundle")
        if (not isinstance(reconstruction_bundle, Mapping)) and hasattr(reconstruction_bundle, "to_dict"):
            reconstruction_bundle = reconstruction_bundle.to_dict()
        if not isinstance(reconstruction_bundle, Mapping):
            reconstruction_bundle_path = external_inputs.get("reconstruction_bundle_json")
            if reconstruction_bundle_path:
                resolved = Path(reconstruction_bundle_path)
                if not resolved.exists() or not resolved.is_file():
                    raise ValueError(f"inference reconstruction_bundle_json does not exist: {resolved}")
                reconstruction_bundle = _load_json_mapping(resolved)
                runtime.payload["reconstruction_bundle"] = dict(reconstruction_bundle)
                input_refs["reconstruction_bundle_json"] = str(resolved)

        template: SyntheticSDFRun | None = None
        selected_template_run_id: str = ""
        if isinstance(dataset, SyntheticSDFDataset) and dataset.runs:
            template = _select_template_run(dataset, requested_template_run_id)
            selected_template_run_id = str(template.run_id)
        elif requested_template_run_id:
            raise ValueError("inference template_run_id requires a dataset input")
        if template is None and not feature_rows:
            raise ValueError("inference stage requires dataset or processed_features for standalone mode")

        model_backend = str(model_state_payload.get("model_backend", "")).strip().lower()
        sparse_rollout_mode = template is not None and (
            isinstance(model, SparseTensorVnModel) or model_backend == "sparse_tensor_checkpoint"
        )
        if sparse_rollout_mode and expected_contract is not None and template is not None:
            recipe_keys = [str(v) for v in expected_contract.get("recipe_keys", [])]
            unresolved = bool(recipe_keys) and all(re.match(r"^recipe_\d+$", key) for key in recipe_keys)
            template_keys = sorted(str(k) for k in template.recipe.keys())
            template_looks_real = any(not re.match(r"^recipe_\d+$", key) for key in template_keys)
            if unresolved and template_looks_real:
                raise ValueError("contract mismatch (recipe_keys unresolved)")
        if expected_contract is not None and feature_rows and not sparse_rollout_mode:
            validate_rows_against_feature_contract(
                rows=feature_rows,
                contract=expected_contract,
                source="inference.feature_contract",
            )

        reference_conditions = _extract_condition_reference(dataset if isinstance(dataset, SyntheticSDFDataset) else None, feature_rows)
        reference_features = _extract_feature_reference(feature_rows)

        calibration_cfg = params.get("calibration")
        calibration_enabled = False
        calibrated_latent: list[float] | None = None
        calibration_method = "none"
        calibration_artifact_path: Path | None = None

        simulation_options: dict[str, object] = {
            "reinit_enabled": bool(reinit_enabled),
            "reinit_every_n": int(reinit_every_n),
            "reinit_iters": int(reinit_iters),
            "reinit_dt": float(reinit_dt),
        }

        if isinstance(calibration_cfg, Mapping) and bool(calibration_cfg.get("enabled", False)):
            calibration_enabled = True
            calibration_method = str(calibration_cfg.get("method", "map")).strip().lower() or "map"
            if template is None:
                warnings.append("calibration skipped: dataset/template is required for SEM projection")
                calibration_enabled = False
            else:
                sem_path = (
                    external_inputs.get("sem_features_json")
                    or external_inputs.get("sem_features_csv")
                    or calibration_cfg.get("sem_features")
                )
                if sem_path is None:
                    warnings.append("calibration skipped: sem_features is not specified")
                    calibration_enabled = False
                else:
                    sem_vec = None
                    try:
                        sem_features = load_sem_features(str(sem_path))
                        sem_vec = [float(value) for value in sem_features.y]
                        if str(sem_path).endswith(".json"):
                            input_refs["sem_features_json"] = str(sem_path)
                        elif str(sem_path).endswith(".csv"):
                            input_refs["sem_features_csv"] = str(sem_path)
                    except (SemFeatureError, ValueError) as exc:
                        warnings.append(f"calibration skipped: invalid sem features ({exc})")
                        calibration_enabled = False

                    if calibration_enabled and sem_vec is not None:
                        latent_dim = max(1, int(calibration_cfg.get("latent_dim", 8)))
                        z_init = [0.0 for _ in range(latent_dim)]
                        prior_mean = [0.0 for _ in range(latent_dim)]
                        prior_std = [1.0 for _ in range(latent_dim)]
                        prior_model = str(calibration_cfg.get("prior_model", "")).strip()
                        if prior_model:
                            prior_kwargs_raw = calibration_cfg.get("prior_kwargs", {})
                            prior_kwargs = (
                                {str(k): v for k, v in prior_kwargs_raw.items()}
                                if isinstance(prior_kwargs_raw, Mapping)
                                else {}
                            )
                            prior_kwargs.setdefault("latent_dim", latent_dim)
                            try:
                                prior = make_shape_prior(prior_model, **prior_kwargs)
                                sample_count = max(8, int(calibration_cfg.get("prior_sample_count", 64)))
                                prior_samples = prior.sample_latent(
                                    num_samples=sample_count,
                                    seed=int(calibration_cfg.get("seed", 0)),
                                )
                                if prior_samples:
                                    z_init = [float(v) for v in prior_samples[0][:latent_dim]]
                                    while len(z_init) < latent_dim:
                                        z_init.append(0.0)
                                    prior_mean = [
                                        fmean(float(sample[idx]) for sample in prior_samples)
                                        for idx in range(latent_dim)
                                    ]
                                    prior_std = []
                                    for idx in range(latent_dim):
                                        mean = float(prior_mean[idx])
                                        var = fmean(
                                            (float(sample[idx]) - mean) ** 2
                                            for sample in prior_samples
                                        )
                                        prior_std.append(max(1e-6, float(var) ** 0.5))
                                    input_refs["prior_model"] = prior_model
                            except Exception as exc:
                                warnings.append(f"calibration prior_model ignored: {exc}")

                        latent_prior_path = external_inputs.get("latent_prior_json")
                        if latent_prior_path:
                            resolved = Path(latent_prior_path)
                            if resolved.exists() and resolved.is_file():
                                prior_payload = _load_latent_prior_payload(
                                    resolved,
                                    latent_dim=latent_dim,
                                )
                                z_init = list(prior_payload["z_init"])
                                prior_mean = list(prior_payload["prior_mean"])
                                prior_std = list(prior_payload["prior_std"])
                                input_refs["latent_prior_json"] = str(resolved)

                        observation_name = str(calibration_cfg.get("observation_model", "baseline"))
                        observation_kwargs_raw = calibration_cfg.get("observation_kwargs", {})
                        observation_kwargs = (
                            {str(k): v for k, v in observation_kwargs_raw.items()}
                            if isinstance(observation_kwargs_raw, Mapping)
                            else {}
                        )
                        try:
                            observation_model = make_observation_model(observation_name, **observation_kwargs)
                        except Exception as exc:
                            warnings.append(f"calibration observation model fallback to baseline: {exc}")
                            observation_model = make_observation_model("baseline")
                        base_conditions = to_float_map(template.recipe)

                        def _simulate_shape(z: Sequence[float]) -> list[list[float]]:
                            cond = _inject_latent(base_conditions, z)
                            local_events: list[dict[str, Any]] = []
                            sim_opts = dict(simulation_options)
                            sim_opts["reinit_log"] = local_events
                            seq = _run_rollout(template, model, cond, simulation_options=sim_opts)
                            for event in local_events:
                                if len(reinit_events) >= 500:
                                    break
                                reinit_events.append(
                                    {
                                        "mode": "calibration",
                                        "step_index": int(event.get("step_index", 0)),
                                        "iters": int(event.get("iters", 0)),
                                        "dt": float(event.get("dt", 0.0)),
                                    }
                                )
                            frame = seq[-1]
                            if hasattr(frame, "tolist"):
                                frame = frame.tolist()
                            return [[float(cell) for cell in row] for row in frame]

                        def _run_map() -> list[float]:
                            map_result = calibrate_latent_map_with_observation(
                                simulate_shape=_simulate_shape,
                                observation_model=observation_model,
                                target_y=sem_vec,
                                z_init=z_init,
                                prior_mean=prior_mean,
                                prior_std=prior_std,
                            )
                            payload = {
                                "method": "map",
                                "z_map": [float(value) for value in map_result.z_map],
                                "y_pred": [float(value) for value in map_result.y_pred],
                                "objective": float(map_result.objective),
                                "feature_loss": float(map_result.feature_loss),
                                "prior_loss": float(map_result.prior_loss),
                                "grad_norm": float(map_result.grad_norm),
                                "iterations": int(map_result.iterations),
                                "converged": bool(map_result.converged),
                            }
                            nonlocal calibration_artifact_path
                            calibration_artifact_path = write_json(stage_dirs["outputs"] / "calibration_map.json", payload)
                            artifacts.append(ArtifactRef(name="calibration_map", path=str(calibration_artifact_path), kind="json"))
                            return [float(value) for value in map_result.z_map]

                        if calibration_method == "sbi":
                            caps = detect_runtime_capabilities()
                            if not caps.sbi:
                                warnings.append(
                                    "calibration sbi fallback to map: optional dependency 'sbi' unavailable"
                                )
                                warnings.append(f"runtime capabilities: {caps.missing_summary()}")
                                calibration_method = "map_fallback"
                                calibrated_latent = _run_map()
                            else:
                                try:
                                    num_sim = max(8, int(calibration_cfg.get("num_sbi_simulations", 32)))
                                    rng = random.Random(int(calibration_cfg.get("seed", 0)))
                                    latent_samples: list[list[float]] = []
                                    observations: list[list[float]] = []
                                    for _ in range(num_sim):
                                        z = [
                                            float(mu) + float(sd) * (rng.random() * 2.0 - 1.0)
                                            for mu, sd in zip(prior_mean, prior_std)
                                        ]
                                        latent_samples.append(z)
                                        observations.append(observation_model.project(_simulate_shape(z)))

                                    estimator = train_latent_posterior_sbi(
                                        latent_samples,
                                        observations,
                                        max_num_epochs=max(20, int(calibration_cfg.get("sbi_max_epochs", 80))),
                                        training_batch_size=max(8, int(calibration_cfg.get("sbi_batch_size", 32))),
                                        device=str(calibration_cfg.get("sbi_device", "cpu")),
                                    )
                                    num_samples = max(1, int(calibration_cfg.get("num_posterior_samples", 32)))
                                    posterior_samples = sample_latent_posterior_sbi(
                                        estimator,
                                        sem_vec,
                                        num_samples=num_samples,
                                        seed=int(calibration_cfg.get("seed", 0)),
                                    )
                                    calibrated_latent = [
                                        fmean([float(sample[idx]) for sample in posterior_samples])
                                        for idx in range(len(posterior_samples[0]))
                                    ]
                                    payload = {
                                        "method": "sbi",
                                        "latent_dim": int(estimator.latent_dim),
                                        "observation_dim": int(estimator.observation_dim),
                                        "num_simulations": int(estimator.num_simulations),
                                        "prior_low": [float(v) for v in estimator.prior_low],
                                        "prior_high": [float(v) for v in estimator.prior_high],
                                        "num_samples": len(posterior_samples),
                                        "samples": [[float(v) for v in row] for row in posterior_samples],
                                        "z_selected_mean": [float(v) for v in calibrated_latent],
                                    }
                                    calibration_artifact_path = write_json(stage_dirs["outputs"] / "calibration_sbi_samples.json", payload)
                                    artifacts.append(
                                        ArtifactRef(
                                            name="calibration_sbi_samples",
                                            path=str(calibration_artifact_path),
                                            kind="json",
                                        )
                                    )
                                except OptionalDependencyUnavailable as exc:
                                    warnings.append(f"calibration sbi fallback to map: {exc}")
                                    warnings.append(
                                        f"runtime capabilities: {detect_runtime_capabilities().missing_summary()}"
                                    )
                                    calibration_method = "map_fallback"
                                    calibrated_latent = _run_map()
                        else:
                            calibrated_latent = _run_map()

        def _apply_latent(candidate: Mapping[str, float]) -> dict[str, float]:
            base = {str(key): float(value) for key, value in candidate.items()}
            if calibrated_latent is None:
                return base
            return _inject_latent(base, calibrated_latent)

        def _evaluate_candidate(candidate: dict[str, float], fidelity: str) -> dict[str, Any]:
            candidate_with_latent = _apply_latent(candidate)
            cond_diag = _condition_vector_diagnostics(model, candidate_with_latent)
            temporal_diag: dict[str, Any] | None = None
            if template is not None:
                local_events: list[dict[str, Any]] = []
                sim_opts = dict(simulation_options)
                sim_opts["reinit_log"] = local_events
                rollout_seq = _run_rollout(template, model, candidate_with_latent, simulation_options=sim_opts)
                objective = frame_mean(rollout_seq[-1])
                temporal_diag = compute_temporal_diagnostics(
                    predicted_phi_t=rollout_seq,
                    reference_phi_t=template.phi_t,
                )
                query_features = None
                if local_events:
                    for event in local_events:
                        if len(reinit_events) >= 1000:
                            break
                        reinit_events.append(
                            {
                                "mode": mode,
                                "step_index": int(event.get("step_index", 0)),
                                "iters": int(event.get("iters", 0)),
                                "dt": float(event.get("dt", 0.0)),
                            }
                        )
            else:
                base_row = feature_rows[0] if feature_rows else {}
                query_row = _condition_feature_projection(base_row, candidate_with_latent)
                objective = float(model.predict(query_row))
                query_features = query_row

            ood = assess_dual_ood(
                query_conditions={str(k): float(v) for k, v in candidate.items()},
                reference_conditions=reference_conditions,
                query_features=query_features,
                reference_features=reference_features if reference_features else None,
                threshold=threshold,
                threshold_profile=ood_reference_profile,
            )
            return {
                "objective": objective,
                "ood_status": ood["status"],
                "ood_distance": ood.get("condition_score"),
                "condition_score": ood.get("condition_score"),
                "feature_score": ood.get("feature_score"),
                "ood_condition": ood.get("condition"),
                "ood_feature": ood.get("feature"),
                "fidelity": fidelity,
                "conditions_used": candidate_with_latent,
                "condition_vector_diagnostics": cond_diag,
                "temporal_diagnostics": temporal_diag,
            }

        if mode == "single":
            cond_cfg = params.get("conditions")
            if isinstance(cond_cfg, Mapping):
                conditions = to_float_map(cond_cfg)
            elif template is not None:
                conditions = to_float_map(template.recipe)
            else:
                conditions = {}

            cond_with_latent = _apply_latent(conditions)
            cond_diag = _condition_vector_diagnostics(model, cond_with_latent)
            rollout_preview_path: Path | None = None
            temporal_diag_single: dict[str, Any] | None = None
            if template is not None:
                local_events: list[dict[str, Any]] = []
                sim_opts = dict(simulation_options)
                sim_opts["reinit_log"] = local_events
                rollout_seq = _run_rollout(template, model, cond_with_latent, simulation_options=sim_opts)
                prediction = {
                    "num_steps": len(rollout_seq),
                    "final_phi_mean": frame_mean(rollout_seq[-1]),
                }
                temporal_diag_single = compute_temporal_diagnostics(
                    predicted_phi_t=rollout_seq,
                    reference_phi_t=template.phi_t,
                )
                rollout_preview_path = write_json(
                    stage_dirs["outputs"] / "inference_single_rollout_preview.json",
                    {
                        "num_steps": len(rollout_seq),
                        "initial_frame": _to_float_nested(rollout_seq[0]),
                        "final_frame": _to_float_nested(rollout_seq[-1]),
                    },
                )
                if local_events:
                    for event in local_events:
                        reinit_events.append(
                            {
                                "mode": "single",
                                "step_index": int(event.get("step_index", 0)),
                                "iters": int(event.get("iters", 0)),
                                "dt": float(event.get("dt", 0.0)),
                            }
                        )
                query_features = None
            else:
                base_row = feature_rows[0] if feature_rows else {}
                query_row = _condition_feature_projection(base_row, cond_with_latent)
                prediction_value = float(model.predict(query_row))
                prediction = {
                    "num_steps": 1,
                    "predicted_target": prediction_value,
                }
                query_features = query_row

            ood = assess_dual_ood(
                query_conditions=conditions,
                reference_conditions=reference_conditions,
                query_features=query_features,
                reference_features=reference_features if reference_features else None,
                threshold=threshold,
                threshold_profile=ood_reference_profile,
            )
            ood_report_path = _write_ood_report(
                stage_dirs=stage_dirs,
                mode="single",
                rows=[
                    {
                        "index": 0,
                        "ood_status": ood["status"],
                        "condition_score": ood.get("condition_score"),
                        "feature_score": ood.get("feature_score"),
                        "ood_condition": ood["condition"],
                        "ood_feature": ood["feature"],
                        "conditions": conditions,
                        "condition_vector_diagnostics": cond_diag,
                    }
                ],
                threshold=threshold,
                extras={
                    "condition_threshold": ood.get("condition_threshold"),
                    "feature_threshold": ood.get("feature_threshold"),
                },
            )
            payload = {
                "mode": "single",
                "conditions": conditions,
                "conditions_used": cond_with_latent,
                "prediction": prediction,
                "template_run_id": None if template is None else selected_template_run_id,
                "ood": ood,
                "ood_report_path": str(ood_report_path),
                "ood_reference_profile": ood_reference_profile,
                "input_refs": input_refs,
                "condition_vector_diagnostics": cond_diag,
                "latent_used": calibrated_latent is not None,
                "calibration_method": calibration_method,
                "rollout_preview_path": None if rollout_preview_path is None else str(rollout_preview_path),
            }
            path = write_json(stage_dirs["outputs"] / "inference_single.json", payload)
            temporal_path: Path | None = None
            temporal_plot_path: Path | None = None
            if temporal_diag_single is not None:
                temporal_rows = [
                    {
                        "index": 0,
                        "template_run_id": selected_template_run_id,
                        **temporal_diag_single,
                    }
                ]
                temporal_path = _write_temporal_diagnostics(
                    stage_dirs=stage_dirs,
                    mode="single",
                    rows=temporal_rows,
                    extras={"template_run_id": selected_template_run_id},
                )
                temporal_plot_path = _plot_temporal_diagnostics_rows(
                    stage_dirs=stage_dirs,
                    mode="single",
                    rows=temporal_rows,
                    enabled=viz_enabled(viz_cfg, "inference.temporal_diagnostics_plot", True),
                    dpi=int(viz_cfg.get("export", {}).get("dpi", 140)) if isinstance(viz_cfg.get("export"), Mapping) else 140,
                    warnings=warnings,
                )
            metrics = {
                "num_steps": float(prediction["num_steps"]),
                "in_domain": 1.0 if bool(ood.get("in_domain", False)) else 0.0,
                "latent_used": 1.0 if calibrated_latent is not None else 0.0,
            }
            if "final_phi_mean" in prediction:
                metrics["final_phi_mean"] = float(prediction["final_phi_mean"])
            if "predicted_target" in prediction:
                metrics["predicted_target"] = float(prediction["predicted_target"])

            artifacts.extend(
                [
                    ArtifactRef(name="inference_single", path=str(path), kind="json"),
                    ArtifactRef(name="inference_single_ood_report", path=str(ood_report_path), kind="json"),
                ]
            )
            if temporal_path is not None:
                artifacts.append(
                    ArtifactRef(name="temporal_diagnostics_single", path=str(temporal_path), kind="json")
                )
            if temporal_plot_path is not None:
                artifacts.append(
                    ArtifactRef(name="temporal_diagnostics_single_plot", path=str(temporal_plot_path), kind="png")
                )
            if rollout_preview_path is not None:
                artifacts.append(
                    ArtifactRef(
                        name="inference_single_rollout_preview",
                        path=str(rollout_preview_path),
                        kind="json",
                    )
                )

            if reinit_enabled or reinit_events:
                reinit_log_path = write_json(
                    stage_dirs["outputs"] / "inference_reinit_log.json",
                    {
                        "enabled": bool(reinit_enabled),
                        "reinit_every_n": int(reinit_every_n),
                        "reinit_iters": int(reinit_iters),
                        "reinit_dt": float(reinit_dt),
                        "num_events": len(reinit_events),
                        "events": reinit_events,
                    },
                )
                artifacts.append(ArtifactRef(name="inference_reinit_log", path=str(reinit_log_path), kind="json"))

            viz_manifest_path = _emit_viz_manifest(
                "single",
                {
                    "temporal_json": None if temporal_path is None else str(temporal_path),
                    "temporal_plot_png": None if temporal_plot_path is None else str(temporal_plot_path),
                    "ood_report_json": str(ood_report_path),
                },
            )
            artifacts.append(ArtifactRef(name="inference_visualization_manifest", path=str(viz_manifest_path), kind="json"))

            return StageResult(
                stage=self.name,
                status="ok",
                metrics=metrics,
                artifacts=artifacts,
                details={**payload, "warnings": warnings, "fallback_reason": "", "visualization_manifest_path": str(viz_manifest_path)},
            )

        if mode == "batch":
            batch_size = max(1, int(params.get("batch_size", 4)))
            conditions_csv = params.get("conditions_csv")
            parsed_rows = _load_condition_rows(str(conditions_csv) if conditions_csv is not None else None)
            if parsed_rows:
                candidates = parsed_rows
                if conditions_csv is not None:
                    input_refs["conditions_csv"] = str(conditions_csv)
            else:
                if template is not None:
                    base = to_float_map(template.recipe)
                elif feature_rows:
                    base = {k[5:]: float(v) for k, v in feature_rows[0].items() if str(k).startswith("feat_")}
                    if not base:
                        base = {}
                else:
                    base = {}
                keys = sorted(base)
                centered = (batch_size - 1) / 2.0
                candidates = []
                for idx in range(batch_size):
                    shift = float(idx) - centered
                    row: dict[str, float] = {}
                    for key in keys:
                        baseline = base[key]
                        delta = max(abs(baseline) * 0.05, 0.02)
                        row[key] = baseline + delta * shift
                    candidates.append(row)

            rows: list[dict[str, Any]] = []
            for index, condition in enumerate(candidates):
                eval_result = _evaluate_candidate(condition, "high")
                rows.append(
                    {
                        "index": index,
                        "objective": float(eval_result["objective"]),
                        "ood_status": eval_result["ood_status"],
                        "ood_distance": eval_result["ood_distance"],
                        "condition_score": eval_result.get("condition_score"),
                        "feature_score": eval_result.get("feature_score"),
                        "ood_condition": eval_result.get("ood_condition"),
                        "ood_feature": eval_result.get("ood_feature"),
                        "conditions": condition,
                        "conditions_used": eval_result["conditions_used"],
                        "condition_vector_diagnostics": eval_result.get("condition_vector_diagnostics"),
                        "temporal_diagnostics": eval_result.get("temporal_diagnostics"),
                    }
                )

            ood_report_path = _write_ood_report(
                stage_dirs=stage_dirs,
                mode="batch",
                rows=[
                    {
                        "index": int(row["index"]),
                        "ood_status": row["ood_status"],
                        "ood_distance": row["ood_distance"],
                        "condition_score": row.get("condition_score"),
                        "feature_score": row.get("feature_score"),
                        "ood_condition": row.get("ood_condition"),
                        "ood_feature": row.get("ood_feature"),
                        "conditions": row["conditions"],
                        "condition_vector_diagnostics": row.get("condition_vector_diagnostics"),
                    }
                    for row in rows
                ],
                threshold=threshold,
                extras={
                    "condition_threshold": (ood_reference_profile or {}).get("condition", {}).get("threshold")
                    if isinstance(ood_reference_profile, Mapping)
                    else None,
                    "feature_threshold": (ood_reference_profile or {}).get("feature", {}).get("threshold")
                    if isinstance(ood_reference_profile, Mapping)
                    else None,
                },
            )
            cond_keys = sorted({key for row in rows for key in row["conditions"].keys()})
            flat_rows = []
            for row in rows:
                flat = {
                    key: row[key]
                    for key in row
                    if key
                    not in {
                        "conditions",
                        "conditions_used",
                        "ood_condition",
                        "ood_feature",
                        "condition_vector_diagnostics",
                        "temporal_diagnostics",
                    }
                }
                for key in cond_keys:
                    flat[f"cond_{key}"] = float(row["conditions"].get(key, 0.0))
                flat_rows.append(flat)
            csv_path = write_csv(
                stage_dirs["outputs"] / "inference_batch.csv",
                flat_rows,
                [
                    "index",
                    "objective",
                    "ood_status",
                    "ood_distance",
                    "condition_score",
                    "feature_score",
                    *[f"cond_{k}" for k in cond_keys],
                ],
            )
            summary = {
                "mode": "batch",
                "template_run_id": None if template is None else selected_template_run_id,
                "num_rows": len(rows),
                "csv_path": str(csv_path),
                "ood_report_path": str(ood_report_path),
                "rows": rows,
                "ood_reference_profile": ood_reference_profile,
                "input_refs": input_refs,
                "condition_vector_diagnostics": rows[0].get("condition_vector_diagnostics") if rows else None,
                "latent_used": calibrated_latent is not None,
                "calibration_method": calibration_method,
            }
            summary_path = write_json(stage_dirs["outputs"] / "inference_batch.json", summary)
            temporal_rows = [
                {
                    "index": int(row["index"]),
                    "template_run_id": selected_template_run_id,
                    **row["temporal_diagnostics"],
                }
                for row in rows
                if isinstance(row.get("temporal_diagnostics"), Mapping)
            ]
            temporal_path: Path | None = None
            temporal_plot_path: Path | None = None
            if temporal_rows:
                temporal_path = _write_temporal_diagnostics(
                    stage_dirs=stage_dirs,
                    mode="batch",
                    rows=temporal_rows,
                    extras={"template_run_id": selected_template_run_id},
                )
                temporal_plot_path = _plot_temporal_diagnostics_rows(
                    stage_dirs=stage_dirs,
                    mode="batch",
                    rows=temporal_rows,
                    enabled=viz_enabled(viz_cfg, "inference.temporal_diagnostics_plot", True),
                    dpi=int(viz_cfg.get("export", {}).get("dpi", 140)) if isinstance(viz_cfg.get("export"), Mapping) else 140,
                    warnings=warnings,
                )
            artifacts.extend(
                [
                    ArtifactRef(name="inference_batch_csv", path=str(csv_path), kind="csv"),
                    ArtifactRef(name="inference_batch_summary", path=str(summary_path), kind="json"),
                    ArtifactRef(name="inference_batch_ood_report", path=str(ood_report_path), kind="json"),
                ]
            )
            if temporal_path is not None:
                artifacts.append(
                    ArtifactRef(name="temporal_diagnostics_batch", path=str(temporal_path), kind="json")
                )
            if temporal_plot_path is not None:
                artifacts.append(
                    ArtifactRef(name="temporal_diagnostics_batch_plot", path=str(temporal_plot_path), kind="png")
                )

            if reinit_enabled or reinit_events:
                reinit_log_path = write_json(
                    stage_dirs["outputs"] / "inference_reinit_log.json",
                    {
                        "enabled": bool(reinit_enabled),
                        "reinit_every_n": int(reinit_every_n),
                        "reinit_iters": int(reinit_iters),
                        "reinit_dt": float(reinit_dt),
                        "num_events": len(reinit_events),
                        "events": reinit_events,
                    },
                )
                artifacts.append(ArtifactRef(name="inference_reinit_log", path=str(reinit_log_path), kind="json"))

            viz_manifest_path = _emit_viz_manifest(
                "batch",
                {
                    "batch_summary_json": str(summary_path),
                    "batch_csv": str(csv_path),
                    "temporal_json": None if temporal_path is None else str(temporal_path),
                    "temporal_plot_png": None if temporal_plot_path is None else str(temporal_plot_path),
                    "ood_report_json": str(ood_report_path),
                },
            )
            artifacts.append(ArtifactRef(name="inference_visualization_manifest", path=str(viz_manifest_path), kind="json"))

            return StageResult(
                stage=self.name,
                status="ok",
                metrics={
                    "batch_size": float(len(rows)),
                    "objective_min": float(min(row["objective"] for row in rows)),
                    "objective_max": float(max(row["objective"] for row in rows)),
                    "latent_used": 1.0 if calibrated_latent is not None else 0.0,
                },
                artifacts=artifacts,
                details={**summary, "warnings": warnings, "fallback_reason": "", "visualization_manifest_path": str(viz_manifest_path)},
            )

        if mode == "optimize":
            range_cfg = params.get("condition_ranges")
            if not isinstance(range_cfg, Mapping) or not range_cfg:
                raise ValueError("inference optimize mode requires inference.condition_ranges")
            ranges = {
                str(key): (float(min(bounds[0], bounds[1])), float(max(bounds[0], bounds[1])))
                for key, bounds in range_cfg.items()
                if isinstance(bounds, Sequence) and not isinstance(bounds, (str, bytes, bytearray)) and len(bounds) == 2
            }
            if not ranges:
                raise ValueError("inference optimize mode requires non-empty 2-element ranges")

            engine = str(params.get("engine", "builtin"))
            strategy = str(params.get("strategy", "random"))
            trials = max(1, int(params.get("trials", 20)))
            seed = int(params.get("seed", 0))

            result = run_optimization_engine(
                engine=engine,
                strategy=strategy,
                ranges=ranges,
                trials=trials,
                seed=seed,
                evaluate=_evaluate_candidate,
                bo_candidate_pool_size=max(6, int(params.get("bo_candidates", 64))),
                mfbo_pool_size=max(3, int(params.get("mfbo_pool_size", 12))),
                mfbo_top_k=max(1, int(params.get("mfbo_top_k", 3))),
                optuna_sampler=str(params.get("optuna_sampler", "tpe")),
            )
            fallback_reason = result.get("fallback_reason")
            if isinstance(fallback_reason, str) and fallback_reason.strip():
                warnings.append(fallback_reason)

            history = result["history"]
            best_so_far: float | None = None
            for row in history:
                current = float(row.get("objective", 0.0))
                if best_so_far is None or current < best_so_far:
                    improvement = 0.0 if best_so_far is None else float(best_so_far - current)
                    best_so_far = current
                else:
                    improvement = 0.0
                row["best_so_far"] = float(best_so_far)
                row["improvement"] = float(improvement)

            feature_importance: dict[str, float] = {}
            raw_importance = result.get("feature_importance")
            if isinstance(raw_importance, Mapping):
                feature_importance = {str(k): float(v) for k, v in raw_importance.items()}
            if not feature_importance:
                feature_importance = _estimate_feature_importance(history)

            cond_keys = sorted({key for row in history for key in row["conditions"].keys()})
            flat_rows = []
            for row in history:
                flat = {
                    key: row[key]
                    for key in row
                    if key
                    not in {
                        "conditions",
                        "conditions_used",
                        "ood_condition",
                        "ood_feature",
                        "condition_vector_diagnostics",
                        "temporal_diagnostics",
                    }
                }
                for key in cond_keys:
                    flat[f"cond_{key}"] = float(row["conditions"].get(key, 0.0))
                flat_rows.append(flat)
            csv_path = write_csv(
                stage_dirs["outputs"] / "inference_optimize_history.csv",
                flat_rows,
                [
                    "trial",
                    "fidelity",
                    "objective",
                    "best_so_far",
                    "improvement",
                    "ood_status",
                    "ood_distance",
                    "condition_score",
                    "feature_score",
                    *[f"cond_{k}" for k in cond_keys],
                ],
            )
            history_json_path = write_json(
                stage_dirs["outputs"] / "inference_optimize_history.json",
                {
                    "mode": "optimize",
                    "requested_engine": result.get("requested_engine"),
                    "resolved_engine": result.get("resolved_engine"),
                    "requested_strategy": result.get("requested_strategy"),
                    "strategy": result.get("strategy"),
                    "fallback_reason": result.get("fallback_reason"),
                    "history": history,
                    "feature_importance": feature_importance,
                    "latent_used": calibrated_latent is not None,
                    "calibration_method": calibration_method,
                },
            )
            ood_report_path = _write_ood_report(
                stage_dirs=stage_dirs,
                mode="optimize",
                rows=[
                    {
                        "trial": int(row["trial"]),
                        "fidelity": row["fidelity"],
                        "ood_status": row.get("ood_status"),
                        "ood_distance": row.get("ood_distance"),
                        "condition_score": row.get("condition_score"),
                        "feature_score": row.get("feature_score"),
                        "ood_condition": row.get("ood_condition"),
                        "ood_feature": row.get("ood_feature"),
                        "conditions": row["conditions"],
                        "condition_vector_diagnostics": row.get("condition_vector_diagnostics"),
                        "temporal_diagnostics": row.get("temporal_diagnostics"),
                    }
                    for row in history
                ],
                threshold=threshold,
                extras={
                    "requested_engine": result.get("requested_engine"),
                    "resolved_engine": result.get("resolved_engine"),
                    "requested_strategy": result.get("requested_strategy"),
                    "strategy": result.get("strategy"),
                    "fallback_reason": result.get("fallback_reason"),
                    "feature_importance": feature_importance,
                    "condition_threshold": (ood_reference_profile or {}).get("condition", {}).get("threshold")
                    if isinstance(ood_reference_profile, Mapping)
                    else None,
                    "feature_threshold": (ood_reference_profile or {}).get("feature", {}).get("threshold")
                    if isinstance(ood_reference_profile, Mapping)
                    else None,
                },
            )
            summary = {
                "mode": "optimize",
                "template_run_id": None if template is None else selected_template_run_id,
                "requested_engine": result.get("requested_engine"),
                "resolved_engine": result.get("resolved_engine"),
                "requested_strategy": result.get("requested_strategy"),
                "strategy": result.get("strategy"),
                "fallback_reason": result.get("fallback_reason"),
                "best_entry": result["best_entry"],
                "fidelity_counts": result["fidelity_counts"],
                "feature_importance": feature_importance,
                "ood_reference_profile": ood_reference_profile,
                "history_csv_path": str(csv_path),
                "history_json_path": str(history_json_path),
                "ood_report_path": str(ood_report_path),
                "input_refs": input_refs,
                "condition_vector_diagnostics": (
                    history[0].get("condition_vector_diagnostics")
                    if history
                    else None
                ),
                "latent_used": calibrated_latent is not None,
                "calibration_method": calibration_method,
            }
            summary_path = write_json(stage_dirs["outputs"] / "inference_optimize_summary.json", summary)
            temporal_rows = [
                {
                    "trial": int(row.get("trial", 0)),
                    "template_run_id": selected_template_run_id,
                    **row["temporal_diagnostics"],
                }
                for row in history
                if isinstance(row.get("temporal_diagnostics"), Mapping)
            ]
            temporal_path: Path | None = None
            temporal_plot_path: Path | None = None
            if temporal_rows:
                temporal_path = _write_temporal_diagnostics(
                    stage_dirs=stage_dirs,
                    mode="optimize",
                    rows=temporal_rows,
                    extras={"template_run_id": selected_template_run_id},
                )
                temporal_plot_path = _plot_temporal_diagnostics_rows(
                    stage_dirs=stage_dirs,
                    mode="optimize",
                    rows=temporal_rows,
                    enabled=viz_enabled(viz_cfg, "inference.temporal_diagnostics_plot", True),
                    dpi=int(viz_cfg.get("export", {}).get("dpi", 140)) if isinstance(viz_cfg.get("export"), Mapping) else 140,
                    warnings=warnings,
                )
            artifacts.extend(
                [
                    ArtifactRef(name="inference_optimize_history", path=str(csv_path), kind="csv"),
                    ArtifactRef(name="inference_optimize_history_json", path=str(history_json_path), kind="json"),
                    ArtifactRef(name="inference_optimize_summary", path=str(summary_path), kind="json"),
                    ArtifactRef(name="inference_optimize_ood_report", path=str(ood_report_path), kind="json"),
                ]
            )
            if temporal_path is not None:
                artifacts.append(
                    ArtifactRef(name="temporal_diagnostics_optimize", path=str(temporal_path), kind="json")
                )
            if temporal_plot_path is not None:
                artifacts.append(
                    ArtifactRef(name="temporal_diagnostics_optimize_plot", path=str(temporal_plot_path), kind="png")
                )

            if reinit_enabled or reinit_events:
                reinit_log_path = write_json(
                    stage_dirs["outputs"] / "inference_reinit_log.json",
                    {
                        "enabled": bool(reinit_enabled),
                        "reinit_every_n": int(reinit_every_n),
                        "reinit_iters": int(reinit_iters),
                        "reinit_dt": float(reinit_dt),
                        "num_events": len(reinit_events),
                        "events": reinit_events,
                    },
                )
                artifacts.append(ArtifactRef(name="inference_reinit_log", path=str(reinit_log_path), kind="json"))

            viz_manifest_path = _emit_viz_manifest(
                "optimize",
                {
                    "optimize_summary_json": str(summary_path),
                    "optimize_history_csv": str(csv_path),
                    "optimize_history_json": str(history_json_path),
                    "temporal_json": None if temporal_path is None else str(temporal_path),
                    "temporal_plot_png": None if temporal_plot_path is None else str(temporal_plot_path),
                    "ood_report_json": str(ood_report_path),
                },
            )
            artifacts.append(ArtifactRef(name="inference_visualization_manifest", path=str(viz_manifest_path), kind="json"))

            return StageResult(
                stage=self.name,
                status="ok",
                metrics={
                    "best_objective": float(result["best_entry"]["objective"]),
                    "num_trials": float(len(history)),
                    "latent_used": 1.0 if calibrated_latent is not None else 0.0,
                },
                artifacts=artifacts,
                details={**summary, "warnings": warnings, "fallback_reason": result.get("fallback_reason"), "visualization_manifest_path": str(viz_manifest_path)},
            )

        raise ValueError(f"unsupported inference mode: {mode}")
