from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from wafer_surrogate.data.io import (
    BackendUnavailableError,
    NarrowBandDataset,
    NarrowBandRun,
    NarrowBandStep,
    synthetic_to_narrow_band_dataset,
    write_hdf5_dataset,
    write_zarr_dataset,
)
from wafer_surrogate.data.mc_logs import PRIV_FEATURE_NAMES, dense_priv_matrix, resolve_privileged_lookup
from wafer_surrogate.data.synthetic import SyntheticSDFDataset, SyntheticSDFRun
from wafer_surrogate.features import make_feature_extractor
from wafer_surrogate.geometry import finite_diff_grad
from wafer_surrogate.core import validate_rows_against_feature_contract
from wafer_surrogate.pipeline.types import ArtifactRef, ReconstructionBundle, StageResult
from wafer_surrogate.pipeline.utils import flatten_frame, write_csv, write_json

NARROW_BAND_BASE_FEATURE_NAMES = ["phi", "grad_norm", "curvature_proxy", "band_distance"]
NARROW_BAND_AUX_FEATURE_NAMES = ["coord_x", "coord_y", "coord_z", "step_index"]


def _shape_features(frame: Any, *, narrow_band_width: float = 0.5) -> dict[str, float]:
    rows = frame.tolist() if hasattr(frame, "tolist") else frame
    values = flatten_frame(rows)
    if not values:
        raise ValueError("frame is empty")

    grad_mag: list[float] = []
    curvature_abs: list[float] = []
    try:
        import numpy as np

        arr = np.asarray(rows, dtype=float)
        grad = finite_diff_grad(arr)
        grad_norm = np.linalg.norm(grad, axis=0)
        grad_mag = [float(v) for v in grad_norm.reshape(-1).tolist()]
        lap = np.zeros_like(arr, dtype=float)
        edge_order = 2 if all(int(size) >= 3 for size in arr.shape) else 1
        for axis in range(arr.ndim):
            second = np.gradient(grad[axis], axis=axis, edge_order=edge_order)
            lap = lap + second
        curvature_abs = [abs(float(v)) for v in lap.reshape(-1).tolist()]
    except Exception:
        grad_mag = [0.0 for _ in values]
        curvature_abs = [0.0 for _ in values]

    return {
        "phi_mean": fmean(values),
        "phi_std": pstdev(values) if len(values) > 1 else 0.0,
        "phi_min": min(values),
        "phi_max": max(values),
        "neg_fraction": sum(1 for value in values if value <= 0.0) / float(len(values)),
        "narrow_band_ratio": sum(1 for value in values if abs(value) <= float(narrow_band_width)) / float(len(values)),
        "grad_abs_mean": fmean(grad_mag),
        "grad_abs_max": max(grad_mag),
        "curvature_proxy_mean": fmean(curvature_abs),
        "curvature_proxy_max": max(curvature_abs),
    }


def _dataset_from_dict(payload: Mapping[str, Any]) -> SyntheticSDFDataset:
    runs_payload = payload.get("runs")
    if not isinstance(runs_payload, list) or not runs_payload:
        raise ValueError("dataset json must include non-empty 'runs'")
    runs: list[SyntheticSDFRun] = []
    for index, run in enumerate(runs_payload):
        if not isinstance(run, Mapping):
            raise ValueError(f"runs[{index}] must be a mapping")
        run_id = str(run.get("run_id", f"run_{index:03d}"))
        dt = float(run.get("dt", 0.1))
        recipe_raw = run.get("recipe", {})
        if not isinstance(recipe_raw, Mapping):
            raise ValueError(f"runs[{index}].recipe must be mapping")
        recipe = {str(k): float(v) for k, v in recipe_raw.items()}
        phi_t_raw = run.get("phi_t")
        if not isinstance(phi_t_raw, list) or not phi_t_raw:
            raise ValueError(f"runs[{index}].phi_t must be non-empty list")
        phi_t = []
        for frame in phi_t_raw:
            if hasattr(frame, "tolist"):
                frame = frame.tolist()
            if not isinstance(frame, list):
                raise ValueError(f"runs[{index}].phi_t frame must be list")
            phi_t.append([[float(cell) for cell in row] for row in frame])
        runs.append(SyntheticSDFRun(run_id=run_id, dt=dt, recipe=recipe, phi_t=phi_t))
    return SyntheticSDFDataset(runs=runs)


def _serialize_narrow_band_dataset(dataset: NarrowBandDataset) -> dict[str, Any]:
    return {
        "runs": [
            {
                "run_id": run.run_id,
                "recipe": [float(value) for value in run.recipe],
                "dt": float(run.dt),
                "steps": [
                    {
                        "coords": [list(row) for row in step.coords],
                        "feat": [list(row) for row in step.feat],
                        "vn_target": [list(row) for row in step.vn_target],
                        "priv": [list(row) for row in step.priv] if step.priv is not None else None,
                    }
                    for step in run.steps
                ],
            }
            for run in dataset.runs
        ]
    }


def _attach_privileged(
    dataset: NarrowBandDataset,
    *,
    lookup: Mapping[tuple[str, int], Mapping[str, float]],
) -> NarrowBandDataset:
    runs: list[NarrowBandRun] = []
    for run in dataset.runs:
        new_steps: list[NarrowBandStep] = []
        for step_idx, step in enumerate(run.steps):
            priv_vector = lookup.get((str(run.run_id), int(step_idx)), {})
            priv_matrix = dense_priv_matrix(priv_vector, len(step.coords))
            new_steps.append(
                NarrowBandStep(
                    coords=[list(row) for row in step.coords],
                    feat=[list(row) for row in step.feat],
                    vn_target=[list(row) for row in step.vn_target],
                    priv=priv_matrix,
                )
            )
        runs.append(
            NarrowBandRun(
                run_id=str(run.run_id),
                recipe=[float(v) for v in run.recipe],
                dt=float(run.dt),
                steps=new_steps,
            )
        )
    return NarrowBandDataset(runs=runs)


def _point_rows_from_narrow_band(
    *,
    dataset: NarrowBandDataset,
    recipe_keys: list[str],
    extractor: Any,
) -> tuple[list[dict[str, float]], list[float], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, float]] = []
    targets: list[float] = []
    row_refs: list[dict[str, Any]] = []
    priv_dim = 0
    feat_dim = 0
    for run_idx, run in enumerate(dataset.runs):
        recipe_map = {
            str(recipe_keys[idx]): float(value)
            for idx, value in enumerate(run.recipe[: len(recipe_keys)])
        }
        base = {str(k): float(v) for k, v in extractor.extract(recipe_map).items()}
        denom = max(1.0, float(len(run.steps) - 1))
        for step_idx, step in enumerate(run.steps):
            n = min(len(step.coords), len(step.feat), len(step.vn_target))
            for point_idx in range(n):
                sample_index = len(rows)
                coord = step.coords[point_idx]
                feat_row = step.feat[point_idx]
                feat_dim = max(feat_dim, len(feat_row))
                priv_row: list[float] = []
                if step.priv is not None and point_idx < len(step.priv):
                    priv_row = [float(v) for v in step.priv[point_idx]]
                    priv_dim = max(priv_dim, len(priv_row))
                row: dict[str, float] = dict(base)
                row["phi"] = float(feat_row[0]) if feat_row else 0.0
                row["grad_norm"] = float(feat_row[1]) if len(feat_row) > 1 else 0.0
                row["curvature_proxy"] = float(feat_row[2]) if len(feat_row) > 2 else 0.0
                row["band_distance"] = float(feat_row[3]) if len(feat_row) > 3 else abs(float(feat_row[0])) if feat_row else 0.0
                for feat_idx, value in enumerate(feat_row):
                    row[f"nb_feat_{feat_idx}"] = float(value)
                row["coord_x"] = float(coord[0])
                row["coord_y"] = float(coord[1])
                row["coord_z"] = float(coord[2])
                row["step_index"] = float(step_idx)
                row["sample_index"] = float(sample_index)
                row["run_index"] = float(run_idx)
                row["run_step"] = float(step_idx)
                row["point_index"] = float(point_idx)
                row["dt"] = float(run.dt)
                row["step_fraction"] = float(step_idx) / denom
                for priv_idx, value in enumerate(priv_row):
                    row[f"priv_{priv_idx}"] = float(value)

                rows.append(row)
                targets.append(float(step.vn_target[point_idx][0]))
                row_refs.append(
                    {
                        "sample_index": sample_index,
                        "run_id": str(run.run_id),
                        "run_index": run_idx,
                        "step_index": step_idx,
                        "point_index": point_idx,
                        "dt": float(run.dt),
                    }
                )
    feature_names = list(NARROW_BAND_BASE_FEATURE_NAMES)
    if feat_dim > len(NARROW_BAND_BASE_FEATURE_NAMES):
        feature_names.extend(f"nb_feat_{idx}" for idx in range(len(NARROW_BAND_BASE_FEATURE_NAMES), feat_dim))
    return rows, targets, row_refs, feature_names


def _validate_point_contract(
    *,
    rows: list[dict[str, float]],
    contract: Mapping[str, Any],
) -> None:
    validate_rows_against_feature_contract(
        rows=rows,
        contract=contract,
        source="feature_contract",
    )


def _write_narrow_band_artifact(
    *,
    dataset: NarrowBandDataset,
    stage_dirs: dict[str, Path],
    backend: str,
) -> tuple[Path, str, list[str]]:
    warnings: list[str] = []
    backend_name = str(backend).strip().lower() or "hdf5"

    if backend_name == "memory":
        path = write_json(stage_dirs["outputs"] / "narrow_band.json", _serialize_narrow_band_dataset(dataset))
        return Path(path), "memory", warnings

    if backend_name in {"hdf5", "h5"}:
        path = stage_dirs["outputs"] / "narrow_band.h5"
        try:
            write_hdf5_dataset(path, dataset)
            return path, "hdf5", warnings
        except BackendUnavailableError as exc:
            warnings.append(f"narrow-band backend fallback from hdf5: {exc}")
            backend_name = "zarr"

    if backend_name == "zarr":
        path = stage_dirs["outputs"] / "narrow_band.zarr"
        try:
            write_zarr_dataset(path, dataset)
            return path, "zarr", warnings
        except BackendUnavailableError as exc:
            warnings.append(f"narrow-band backend fallback from zarr: {exc}")

    path = write_json(stage_dirs["outputs"] / "narrow_band.json", _serialize_narrow_band_dataset(dataset))
    warnings.append("narrow-band backend fallback to memory json")
    return Path(path), "memory", warnings


class FeaturizationStage:
    name = "featurization"

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
        params = runtime.stage_params("featurization")
        extractor_name = str(params.get("extractor", "identity"))
        extractor_kwargs = params.get("extractor_kwargs", {})
        if not isinstance(extractor_kwargs, dict):
            extractor_kwargs = {}
        extractor = make_feature_extractor(extractor_name, **extractor_kwargs)

        external_inputs = self._stage_external_inputs(runtime)
        dataset = runtime.payload.get("dataset_clean") or runtime.payload.get("dataset_raw")
        input_refs: dict[str, str] = {}
        warnings: list[str] = []
        if not isinstance(dataset, SyntheticSDFDataset):
            dataset_path = (
                external_inputs.get("raw_dataset_json")
                or external_inputs.get("dataset_json")
                or external_inputs.get("dataset")
            )
            if dataset_path:
                resolved = Path(dataset_path)
                if not resolved.exists() or not resolved.is_file():
                    raise ValueError(f"featurization dataset_json does not exist: {resolved}")
                with resolved.open("r", encoding="utf-8") as fp:
                    payload = json.load(fp)
                if not isinstance(payload, Mapping):
                    raise ValueError("featurization dataset_json must contain a mapping")
                dataset = _dataset_from_dict(payload)
                input_refs["dataset_json"] = str(resolved)
        if not isinstance(dataset, SyntheticSDFDataset):
            raise ValueError("featurization stage requires dataset_clean/dataset_raw or external_inputs.dataset_json")

        target_mode = str(params.get("target_mode", "frame_mean_delta")).strip().lower() or "frame_mean_delta"
        band_width = float(params.get("band_width", params.get("narrow_band_width", 0.5)))
        min_grad_norm = float(params.get("min_grad_norm", 1e-6))
        include_terminal_step_target = bool(params.get("include_terminal_step_target", False))
        emit_priv = bool(params.get("emit_priv", False))
        nb_backend = str(params.get("nb_backend", "hdf5"))
        priv_source = str(params.get("priv_source", "auto")).strip().lower() or "auto"

        narrow_band_dataset: NarrowBandDataset | None = None
        narrow_band_path: Path | None = None
        narrow_band_backend: str | None = None
        narrow_band_manifest_path: Path | None = None
        recipe_keys = sorted(dataset.runs[0].recipe.keys()) if dataset.runs else []
        used_priv_source: str | None = None

        if target_mode == "vn_narrow_band":
            narrow_band_dataset = synthetic_to_narrow_band_dataset(
                dataset,
                band_width=band_width,
                min_grad_norm=min_grad_norm,
                include_terminal_step_target=include_terminal_step_target,
            )

            total_points = sum(len(step.coords) for run in narrow_band_dataset.runs for step in run.steps)
            if emit_priv:
                lookup, used_priv_source, priv_warnings, priv_refs = resolve_privileged_lookup(
                    dataset=dataset,
                    source=priv_source,
                    mc_log_jsonl=external_inputs.get("mc_log_jsonl"),
                    mc_log_h5=external_inputs.get("mc_log_h5"),
                )
                warnings.extend(priv_warnings)
                input_refs.update(priv_refs)
                narrow_band_dataset = _attach_privileged(narrow_band_dataset, lookup=lookup)
            else:
                used_priv_source = "disabled"

            narrow_band_path, narrow_band_backend, backend_warnings = _write_narrow_band_artifact(
                dataset=narrow_band_dataset,
                stage_dirs=stage_dirs,
                backend=nb_backend,
            )
            warnings.extend(backend_warnings)
            runtime.payload["narrow_band_dataset"] = narrow_band_dataset
            runtime.payload["narrow_band_backend"] = narrow_band_backend
            runtime.payload["narrow_band_path"] = str(narrow_band_path)
            runtime.payload["narrow_band_priv_source"] = used_priv_source

            nb_manifest = {
                "schema_version": "1",
                "target_mode": "vn_narrow_band",
                "backend": narrow_band_backend,
                "dataset_path": str(narrow_band_path),
                "recipe_keys": list(recipe_keys),
                "band_width": float(band_width),
                "min_grad_norm": float(min_grad_norm),
                "terminal_target_policy": "include" if include_terminal_step_target else "exclude",
                "emit_priv": bool(emit_priv),
                "priv_source": str(used_priv_source or "none"),
                "priv_feature_names": list(PRIV_FEATURE_NAMES),
                "num_runs": len(narrow_band_dataset.runs),
                "num_steps_total": sum(len(run.steps) for run in narrow_band_dataset.runs),
                "num_points_total": int(total_points),
                "input_refs": dict(input_refs),
            }
            narrow_band_manifest_path = write_json(stage_dirs["outputs"] / "narrow_band_manifest.json", nb_manifest)
            runtime.payload["narrow_band_manifest"] = nb_manifest

            point_rows, point_targets, point_refs, point_feature_names = _point_rows_from_narrow_band(
                dataset=narrow_band_dataset,
                recipe_keys=recipe_keys,
                extractor=extractor,
            )
            if not point_rows or not point_targets:
                raise ValueError("vn_narrow_band featurization produced no point-level samples")
            point_manifest = {
                "schema_version": "1",
                "target_mode": "vn_narrow_band",
                "num_rows": int(len(point_rows)),
                "num_targets": int(len(point_targets)),
                "sample_ref_fields": ["run_id", "step_index", "point_index", "sample_index"],
                "feature_contract": {
                    "recipe_keys": list(recipe_keys),
                    "feature_names": [*point_feature_names, *NARROW_BAND_AUX_FEATURE_NAMES],
                    "cond_dim": len(recipe_keys),
                    "feat_dim": len(point_feature_names) + len(NARROW_BAND_AUX_FEATURE_NAMES),
                    "band_width": float(band_width),
                    "min_grad_norm": float(min_grad_norm),
                },
                "terminal_target_policy": "include" if include_terminal_step_target else "exclude",
                "input_refs": dict(input_refs),
            }
            _validate_point_contract(rows=point_rows, contract=point_manifest["feature_contract"])
            point_manifest_path = write_json(stage_dirs["outputs"] / "point_level_manifest.json", point_manifest)
            runtime.payload["point_level_manifest"] = point_manifest
        else:
            point_rows = []
            point_targets = []
            point_refs = []
            point_manifest_path = None

        rows: list[dict[str, float]] = []
        targets: list[float] = []
        row_refs: list[dict[str, Any]] = []
        narrow_band_width = float(params.get("narrow_band_width", params.get("band_width", 0.5)))
        if target_mode == "vn_narrow_band":
            rows = point_rows
            targets = point_targets
            row_refs = point_refs
        else:
            for run_idx, run in enumerate(dataset.runs):
                base = {str(k): float(v) for k, v in extractor.extract(run.recipe).items()}
                steps = max(1, len(run.phi_t) - 1)
                for step_idx in range(len(run.phi_t) - 1):
                    sample_index = len(rows)
                    prev_frame = run.phi_t[step_idx]
                    next_frame = run.phi_t[step_idx + 1]
                    prev_mean = fmean(flatten_frame(prev_frame))
                    next_mean = fmean(flatten_frame(next_frame))

                    feature_row = dict(base)
                    feature_row.update(_shape_features(prev_frame, narrow_band_width=narrow_band_width))
                    feature_row["step_fraction"] = float(step_idx) / float(steps)
                    feature_row["step_remaining_fraction"] = float(steps - step_idx) / float(steps)
                    feature_row["dt"] = float(run.dt)
                    feature_row["sample_index"] = float(sample_index)
                    feature_row["run_index"] = float(run_idx)
                    feature_row["run_step"] = float(step_idx)
                    feature_row["phi_mean_t"] = prev_mean
                    feature_row["phi_mean_t_plus_1"] = next_mean
                    if step_idx > 0:
                        prev_prev_mean = fmean(flatten_frame(run.phi_t[step_idx - 1]))
                        feature_row["phi_mean_lag1"] = prev_mean - prev_prev_mean
                        feature_row["target_lag1"] = (prev_prev_mean - prev_mean) / float(run.dt)
                    else:
                        feature_row["phi_mean_lag1"] = 0.0
                        feature_row["target_lag1"] = 0.0

                    rows.append(feature_row)
                    targets.append((prev_mean - next_mean) / float(run.dt))
                    row_refs.append(
                        {
                            "sample_index": sample_index,
                            "run_id": str(run.run_id),
                            "run_index": run_idx,
                            "step_index": step_idx,
                            "dt": float(run.dt),
                        }
                    )

        if not rows or not targets:
            raise ValueError("featurization produced no samples")

        runtime.payload["features"] = rows
        runtime.payload["targets"] = targets
        fieldnames = sorted({key for row in rows for key in row.keys()})
        for row in rows:
            for key in fieldnames:
                row.setdefault(key, 0.0)
        rows_path = write_csv(stage_dirs["outputs"] / "features.csv", rows, fieldnames)
        target_rows = [{"index": idx, "target": float(value)} for idx, value in enumerate(targets)]
        targets_path = write_csv(stage_dirs["outputs"] / "targets.csv", target_rows, ["index", "target"])
        total_steps = sum(max(0, len(run.phi_t) - 1) for run in dataset.runs)

        target_formula = "(mean(phi_t)-mean(phi_t_plus_1))/dt"
        if target_mode == "vn_narrow_band":
            target_formula = "(phi_t-phi_t_plus_1)/(dt*|grad(phi_t)|) at narrow-band points"

        inverse_mapping: dict[str, Any] = {
            "schema_version": "2",
            "sample_ref_fields": ["sample_index", "run_id", "run_index", "step_index", "point_index"] if target_mode == "vn_narrow_band" else ["sample_index", "run_id", "run_index", "step_index"],
            "feature_fields": fieldnames,
            "target_name": "target",
            "target_formula": target_formula,
            "target_mode": target_mode,
            "time_features": ["step_fraction", "step_remaining_fraction", "phi_mean_lag1", "target_lag1"],
        }
        if narrow_band_manifest_path is not None:
            inverse_mapping["narrow_band_manifest_path"] = str(narrow_band_manifest_path)
            inverse_mapping["terminal_target_policy"] = "include" if include_terminal_step_target else "exclude"
        if point_manifest_path is not None:
            inverse_mapping["point_level_manifest_path"] = str(point_manifest_path)

        bundle = ReconstructionBundle(
            id=f"{runtime.run_dir.name}:featurization",
            payload_path=str(rows_path),
            target_path=str(targets_path),
            metrics={
                "schema_version": 2.0,
                "num_runs": float(len(dataset.runs)),
                "num_steps": float(total_steps),
                "num_rows": float(len(rows)),
            },
            row_refs=row_refs,
            inverse_mapping=inverse_mapping,
            post_inference_hooks=[
                "attach_predictions_to_sample_refs",
                "reconstruct_per_run_time_series",
            ],
        )
        runtime.payload["reconstruction_bundle"] = bundle
        bundle_path = write_json(stage_dirs["outputs"] / "reconstruction_bundle.json", bundle.to_dict())

        artifacts = [
            ArtifactRef(name="features", path=str(rows_path), kind="csv"),
            ArtifactRef(name="targets", path=str(targets_path), kind="csv"),
            ArtifactRef(name="reconstruction_bundle", path=str(bundle_path), kind="json"),
        ]
        if narrow_band_path is not None:
            artifacts.append(
                ArtifactRef(
                    name="narrow_band_dataset",
                    path=str(narrow_band_path),
                    kind=("hdf5" if str(narrow_band_path).endswith(".h5") else ("zarr" if str(narrow_band_path).endswith(".zarr") else "json")),
                )
            )
        if narrow_band_manifest_path is not None:
            artifacts.append(ArtifactRef(name="narrow_band_manifest", path=str(narrow_band_manifest_path), kind="json"))
        if point_manifest_path is not None:
            artifacts.append(ArtifactRef(name="point_level_manifest", path=str(point_manifest_path), kind="json"))

        details: dict[str, Any] = {
            "extractor": extractor_name,
            "num_rows": len(rows),
            "num_targets": len(targets),
            "target_mode": target_mode,
            "input_refs": input_refs,
            "warnings": warnings,
        }
        if narrow_band_backend is not None:
            details["narrow_band_backend"] = narrow_band_backend
            details["priv_source"] = used_priv_source

        return StageResult(
            stage=self.name,
            status="ok",
            metrics={
                "num_feature_rows": float(len(rows)),
                "num_feature_cols": float(len(fieldnames)),
                "target_mean": fmean(targets),
            },
            artifacts=artifacts,
            details=details,
        )
