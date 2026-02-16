from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wafer_surrogate.pipeline.types import ArtifactRef, StageResult
from wafer_surrogate.pipeline.utils import write_csv, write_json
from wafer_surrogate.preprocess import (
    build_preprocess_pipeline,
    fit_feature_scaler,
    fit_pca_projector,
    fit_quantile_feature_transformer,
    fit_robust_feature_scaler,
    fit_target_transform,
)


class PreprocessingStage:
    name = "preprocessing"

    def _signed_log1p_row(self, row: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in row.items():
            vv = float(value)
            out[str(key)] = math.copysign(math.log1p(abs(vv)), vv)
        return out

    def _inverse_signed_log1p_row(self, row: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in row.items():
            vv = float(value)
            out[str(key)] = math.copysign(math.expm1(abs(vv)), vv)
        return out

    def _feature_contract_hash(self, payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _build_preprocess_report(
        self,
        *,
        raw_features: list[dict[str, float]],
        reconstructed_features: list[dict[str, float]] | None,
        raw_targets: list[float],
        reconstructed_targets: list[float] | None,
    ) -> dict[str, Any]:
        keys = sorted({str(key) for row in raw_features for key in row.keys()})
        n_rows = max(1, len(raw_features))
        feature_stats: dict[str, Any] = {}
        for key in keys:
            values = [float(row.get(key, 0.0)) for row in raw_features]
            if values:
                mean = sum(values) / float(len(values))
                var = sum((float(v) - mean) ** 2 for v in values) / float(len(values))
                std = math.sqrt(var)
            else:
                mean = 0.0
                var = 0.0
                std = 0.0
            missing = sum(1 for row in raw_features if key not in row)
            outliers = 0
            if std > 1e-12:
                outliers = sum(1 for v in values if abs((float(v) - mean) / std) > 3.0)
            feature_stats[key] = {
                "mean": float(mean),
                "variance": float(var),
                "missing_rate": float(missing) / float(n_rows),
                "outlier_rate": float(outliers) / float(max(1, len(values))),
            }

        reconstruction = {"inverse_ready": False, "feature_mae": None, "target_mae": None}
        if reconstructed_features is not None and len(reconstructed_features) == len(raw_features) and raw_features:
            err_total = 0.0
            err_count = 0
            for raw_row, rec_row in zip(raw_features, reconstructed_features):
                for key in keys:
                    err_total += abs(float(raw_row.get(key, 0.0)) - float(rec_row.get(key, 0.0)))
                    err_count += 1
            reconstruction["feature_mae"] = float(err_total / float(max(1, err_count)))
            reconstruction["inverse_ready"] = True
        if reconstructed_targets is not None and len(reconstructed_targets) == len(raw_targets) and raw_targets:
            target_err = sum(
                abs(float(lhs) - float(rhs))
                for lhs, rhs in zip(raw_targets, reconstructed_targets)
            ) / float(len(raw_targets))
            reconstruction["target_mae"] = float(target_err)
            reconstruction["inverse_ready"] = bool(reconstruction["inverse_ready"])

        return {
            "schema_version": "1",
            "num_rows": len(raw_features),
            "num_target_rows": len(raw_targets),
            "feature_stats": feature_stats,
            "reconstruction_error": reconstruction,
        }

    def _validate_preprocess_bundle(self, bundle: Mapping[str, Any]) -> None:
        required = {
            "schema_version",
            "steps",
            "feature_transform",
            "feature_log1p",
            "normalization",
            "feature_scaler",
            "target_transform",
            "feature_contract_hash",
            "inverse_ready",
            "inverse_metadata",
            "inverse_mapping",
        }
        missing = sorted(required.difference(set(bundle.keys())))
        if missing:
            raise ValueError(f"preprocessing bundle missing required keys: {missing}")
        if not isinstance(bundle.get("normalization"), Mapping):
            raise ValueError("preprocessing bundle normalization must be a mapping")
        if not isinstance(bundle.get("inverse_metadata"), Mapping):
            raise ValueError("preprocessing bundle inverse_metadata must be a mapping")
        if not isinstance(bundle.get("inverse_mapping"), Mapping):
            raise ValueError("preprocessing bundle inverse_mapping must be a mapping")

    def _validate_preprocess_report(self, report: Mapping[str, Any]) -> None:
        required = {"schema_version", "num_rows", "num_target_rows", "feature_stats", "reconstruction_error"}
        missing = sorted(required.difference(set(report.keys())))
        if missing:
            raise ValueError(f"preprocessing report missing required keys: {missing}")
        feature_stats = report.get("feature_stats")
        if not isinstance(feature_stats, Mapping):
            raise ValueError("preprocessing report feature_stats must be a mapping")
        for key, value in feature_stats.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"preprocessing report feature_stats[{key}] must be a mapping")
            for stat_key in ("mean", "variance", "missing_rate", "outlier_rate"):
                if stat_key not in value:
                    raise ValueError(f"preprocessing report feature_stats[{key}] missing '{stat_key}'")
        reconstruction_error = report.get("reconstruction_error")
        if not isinstance(reconstruction_error, Mapping):
            raise ValueError("preprocessing report reconstruction_error must be a mapping")
        if "inverse_ready" not in reconstruction_error:
            raise ValueError("preprocessing report reconstruction_error missing 'inverse_ready'")

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
        if not path.exists() or not path.is_file():
            raise ValueError(f"preprocessing features_csv does not exist: {path}")
        import csv

        with path.open("r", encoding="utf-8", newline="") as fp:
            rows = [dict(row) for row in csv.DictReader(fp)]
        out: list[dict[str, float]] = []
        for row in rows:
            out.append({str(key): float(value) for key, value in row.items()})
        return out

    def _read_target_values(self, path: Path) -> list[float]:
        if not path.exists() or not path.is_file():
            raise ValueError(f"preprocessing targets_csv does not exist: {path}")
        import csv

        with path.open("r", encoding="utf-8", newline="") as fp:
            rows = [dict(row) for row in csv.DictReader(fp)]
        if not rows:
            return []
        key = "target"
        if key not in rows[0]:
            candidate_keys = [name for name in rows[0].keys() if name not in {"index", "sample_index"}]
            if not candidate_keys:
                raise ValueError(f"preprocessing targets_csv has no usable numeric column: {path}")
            key = candidate_keys[0]
        return [float(row[key]) for row in rows]

    def _read_json_mapping(self, path: Path) -> dict[str, Any]:
        if not path.exists() or not path.is_file():
            raise ValueError(f"preprocessing reconstruction_bundle_json does not exist: {path}")
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, dict):
            raise ValueError(f"preprocessing reconstruction_bundle_json must be a mapping: {path}")
        return payload

    def run(self, runtime: Any, stage_dirs: dict[str, Path]) -> StageResult:
        params = runtime.stage_params("preprocessing")
        external_inputs = self._stage_external_inputs(runtime)
        warnings: list[str] = []

        features = runtime.payload.get("features")
        targets = runtime.payload.get("targets")
        reconstruction_bundle = runtime.payload.get("reconstruction_bundle")

        source_artifacts: dict[str, str] = {}
        if not isinstance(features, list) or not features:
            source_path = external_inputs.get("features_csv") or external_inputs.get("features")
            if source_path:
                resolved = Path(source_path)
                features = self._read_feature_rows(resolved)
                source_artifacts["features_csv"] = str(resolved)
        if not isinstance(targets, list) or not targets:
            source_path = external_inputs.get("targets_csv") or external_inputs.get("targets")
            if source_path:
                resolved = Path(source_path)
                targets = self._read_target_values(resolved)
                source_artifacts["targets_csv"] = str(resolved)
        if (not isinstance(reconstruction_bundle, dict)) and hasattr(reconstruction_bundle, "to_dict"):
            reconstruction_bundle = reconstruction_bundle.to_dict()
        if not isinstance(reconstruction_bundle, dict):
            source_path = external_inputs.get("reconstruction_bundle_json") or external_inputs.get("reconstruction_bundle")
            if source_path:
                resolved = Path(source_path)
                reconstruction_bundle = self._read_json_mapping(resolved)
                source_artifacts["reconstruction_bundle_json"] = str(resolved)

        if not isinstance(features, list) or not features:
            raise ValueError("preprocessing stage requires features from featurization or external_inputs.features_csv")
        if not isinstance(targets, list) or not targets:
            raise ValueError("preprocessing stage requires targets from featurization or external_inputs.targets_csv")
        if isinstance(reconstruction_bundle, dict):
            runtime.payload["reconstruction_bundle"] = reconstruction_bundle

        step_specs_cfg = params.get("steps", ["identity"])
        step_specs: list[Any] = []
        step_descriptors: list[dict[str, Any]] = []
        if isinstance(step_specs_cfg, list):
            for item in step_specs_cfg:
                if isinstance(item, str):
                    step_specs.append(item)
                    step_descriptors.append({"name": item, "params": {}})
                elif isinstance(item, dict):
                    name = str(item.get("name", "identity"))
                    kwargs = item.get("params", {})
                    if not isinstance(kwargs, dict):
                        kwargs = {}
                    frozen_kwargs = {str(key): value for key, value in kwargs.items()}
                    step_specs.append((name, frozen_kwargs))
                    step_descriptors.append({"name": name, "params": frozen_kwargs})
        if not step_specs:
            step_specs = ["identity"]
            step_descriptors = [{"name": "identity", "params": {}}]

        pipeline = build_preprocess_pipeline(step_specs)
        transformed_features = [pipeline.transform({str(k): float(v) for k, v in row.items()}) for row in features]

        feature_log1p = bool(params.get("feature_log1p", False))
        transformed_rows = (
            [self._signed_log1p_row(row) for row in transformed_features]
            if feature_log1p
            else [{str(k): float(v) for k, v in row.items()} for row in transformed_features]
        )

        feature_transform = str(params.get("feature_transform", "")).strip().lower()
        if not feature_transform:
            feature_normalize = bool(params.get("feature_normalize", params.get("normalize", True)))
            feature_transform = "standard" if feature_normalize else "none"

        scaler_meta: dict[str, Any] = {}
        inverse_feature_row: Any | None = None
        if feature_transform in {"none", "identity"}:
            feature_scaler = fit_feature_scaler([])
            processed_features = [{str(k): float(v) for k, v in row.items()} for row in transformed_rows]
            scaler_meta = {"mode": "none"}
            inverse_feature_row = feature_scaler.inverse_row
        elif feature_transform in {"standard", "zscore"}:
            feature_scaler = fit_feature_scaler(transformed_rows)
            processed_features = [feature_scaler.transform_row(row) for row in transformed_rows]
            scaler_meta = feature_scaler.to_dict()
            inverse_feature_row = feature_scaler.inverse_row
        elif feature_transform == "robust":
            robust = fit_robust_feature_scaler(transformed_rows)
            feature_scaler = fit_feature_scaler([])
            processed_features = [robust.transform_row(row) for row in transformed_rows]
            scaler_meta = robust.to_dict()
            inverse_feature_row = robust.inverse_row
        elif feature_transform == "quantile":
            quant = fit_quantile_feature_transformer(transformed_rows)
            feature_scaler = fit_feature_scaler([])
            processed_features = [quant.transform_row(row) for row in transformed_rows]
            scaler_meta = quant.to_dict()
            inverse_feature_row = quant.inverse_row
        elif feature_transform == "pca":
            n_components = int(params.get("pca_components", 3))
            try:
                pca = fit_pca_projector(transformed_rows, n_components=n_components)
                feature_scaler = fit_feature_scaler([])
                processed_features = [pca.transform_row(row) for row in transformed_rows]
                scaler_meta = pca.to_dict()
                inverse_feature_row = pca.inverse_row
            except Exception as exc:
                warnings.append(f"pca preprocessing fallback to standard: {exc}")
                feature_scaler = fit_feature_scaler(transformed_rows)
                processed_features = [feature_scaler.transform_row(row) for row in transformed_rows]
                scaler_meta = feature_scaler.to_dict()
                feature_transform = "standard"
                inverse_feature_row = feature_scaler.inverse_row
        else:
            raise ValueError(f"unsupported preprocessing feature_transform: {feature_transform}")

        target_transform_mode = str(params.get("target_transform", "identity"))
        target_transformer = fit_target_transform([float(v) for v in targets], mode=target_transform_mode)
        processed_targets = target_transformer.transform([float(v) for v in targets])

        runtime.payload["processed_features"] = processed_features
        runtime.payload["processed_targets"] = processed_targets

        source_inverse_mapping: dict[str, Any] = {}
        source_row_refs: list[Any] = []
        source_hooks: list[str] = []
        source_bundle_id: str | None = None
        if isinstance(reconstruction_bundle, dict):
            source_bundle_id = str(reconstruction_bundle.get("id")) if reconstruction_bundle.get("id") is not None else None
            candidate_inverse = reconstruction_bundle.get("inverse_mapping")
            if isinstance(candidate_inverse, dict):
                source_inverse_mapping = dict(candidate_inverse)
            candidate_row_refs = reconstruction_bundle.get("row_refs")
            if isinstance(candidate_row_refs, list):
                source_row_refs = [dict(row) for row in candidate_row_refs if isinstance(row, dict)]
            candidate_hooks = reconstruction_bundle.get("post_inference_hooks")
            if isinstance(candidate_hooks, list):
                source_hooks = [str(value) for value in candidate_hooks]

        fieldnames = sorted({key for row in processed_features for key in row.keys()})
        source_feature_fields = source_inverse_mapping.get("feature_fields", [])
        if not isinstance(source_feature_fields, list):
            source_feature_fields = []
        contract_payload = {
            "source_feature_fields": [str(v) for v in source_feature_fields],
            "processed_feature_fields": [str(v) for v in fieldnames],
            "source_sample_ref_fields": source_inverse_mapping.get("sample_ref_fields", []),
            "target_mode": source_inverse_mapping.get("target_mode", ""),
        }
        feature_contract_hash = self._feature_contract_hash(contract_payload)
        inverse_ready = (
            inverse_feature_row is not None
            and all(str(step.get("name", "identity")) == "identity" for step in step_descriptors)
        )
        inverse_metadata = {
            "schema_version": "2",
            "target_name": "target",
            "target_transform": target_transformer.to_dict(),
            "processed_feature_fields": fieldnames,
            "feature_normalization_applied": feature_transform not in {"none", "identity"},
            "feature_scaler": scaler_meta,
            "feature_transform": feature_transform,
            "feature_log1p": feature_log1p,
            "steps": step_descriptors,
            "source_bundle_id": source_bundle_id,
            "source_sample_ref_fields": source_inverse_mapping.get("sample_ref_fields", []),
            "source_row_ref_count": len(source_row_refs),
            "source_artifacts": source_artifacts,
            "feature_contract_hash": feature_contract_hash,
            "inverse_ready": bool(inverse_ready),
            "inverse_transform_contract": {
                "features": "apply feature_scaler inverse_row then reverse preprocessing steps if reversible",
                "targets": "apply target_transform inverse",
            },
        }
        runtime.payload["preprocess_bundle"] = {
            "schema_version": "2",
            "steps": step_descriptors,
            "feature_transform": feature_transform,
            "feature_log1p": feature_log1p,
            "normalization": {
                key: {
                    "mean": float(value),
                    "std": float(feature_scaler.stds.get(key, 1.0)),
                }
                for key, value in feature_scaler.means.items()
            },
            "feature_scaler": scaler_meta,
            "target_transform": target_transformer.to_dict(),
            "feature_contract_hash": feature_contract_hash,
            "inverse_ready": bool(inverse_ready),
            "inverse_metadata": inverse_metadata,
            "inverse_mapping": dict(inverse_metadata),
        }
        if source_row_refs:
            runtime.payload["preprocess_bundle"]["row_refs"] = source_row_refs
        if source_hooks:
            runtime.payload["preprocess_bundle"]["post_inference_hooks"] = source_hooks
        self._validate_preprocess_bundle(runtime.payload["preprocess_bundle"])

        for row in processed_features:
            for key in fieldnames:
                row.setdefault(key, 0.0)
        processed_path = write_csv(stage_dirs["outputs"] / "processed_features.csv", processed_features, fieldnames)
        target_rows = [{"index": idx, "target": float(value)} for idx, value in enumerate(processed_targets)]
        targets_path = write_csv(stage_dirs["outputs"] / "processed_targets.csv", target_rows, ["index", "target"])
        bundle_path = write_json(stage_dirs["outputs"] / "preprocess_bundle.json", dict(runtime.payload["preprocess_bundle"]))
        reconstructed_features: list[dict[str, float]] | None = None
        if inverse_ready and inverse_feature_row is not None:
            reconstructed_features = []
            for row in processed_features:
                restored = inverse_feature_row(row)
                if feature_log1p:
                    restored = self._inverse_signed_log1p_row(restored)
                reconstructed_features.append({str(k): float(v) for k, v in restored.items()})
        reconstructed_targets = target_transformer.inverse(processed_targets)
        report = self._build_preprocess_report(
            raw_features=transformed_features,
            reconstructed_features=reconstructed_features,
            raw_targets=[float(v) for v in targets],
            reconstructed_targets=reconstructed_targets,
        )
        self._validate_preprocess_report(report)
        report_path = write_json(stage_dirs["outputs"] / "preprocess_report.json", report)

        return StageResult(
            stage=self.name,
            status="ok",
            metrics={
                "num_rows": float(len(processed_features)),
                "num_cols": float(len(fieldnames)),
                "feature_normalized": 1.0 if feature_transform not in {"none", "identity"} else 0.0,
                "target_standardized": 1.0 if target_transformer.mode == "standard" else 0.0,
            },
            artifacts=[
                ArtifactRef(name="processed_features", path=str(processed_path), kind="csv"),
                ArtifactRef(name="processed_targets", path=str(targets_path), kind="csv"),
                ArtifactRef(name="preprocess_bundle", path=str(bundle_path), kind="json"),
                ArtifactRef(name="preprocess_report", path=str(report_path), kind="json"),
            ],
            details={
                "steps": step_descriptors,
                "feature_normalize": feature_transform not in {"none", "identity"},
                "feature_transform": feature_transform,
                "feature_log1p": feature_log1p,
                "target_transform": target_transformer.mode,
                "inverse_ready": bool(inverse_ready),
                "used_external_inputs": bool(source_artifacts),
                "input_refs": source_artifacts,
                "warnings": warnings,
            },
        )
