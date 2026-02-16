from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from wafer_surrogate.data.synthetic import SyntheticSDFDataset, SyntheticSDFRun
from wafer_surrogate.pipeline.types import ArtifactRef, StageResult
from wafer_surrogate.pipeline.utils import write_json


def _parse_bounds_map(value: object) -> dict[str, tuple[float, float]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, tuple[float, float]] = {}
    for key, bounds in value.items():
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            lo = float(min(bounds[0], bounds[1]))
            hi = float(max(bounds[0], bounds[1]))
            out[str(key)] = (lo, hi)
    return out


def _resolve_normalization_ranges(
    *,
    runs: list[Any],
    explicit_ranges: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    if explicit_ranges:
        return dict(explicit_ranges)
    values_by_key: dict[str, list[float]] = {}
    for run in runs:
        for key, value in run.recipe.items():
            float_value = float(value)
            if float_value != float_value:  # NaN
                continue
            values_by_key.setdefault(str(key), []).append(float_value)
    out: dict[str, tuple[float, float]] = {}
    for key, values in values_by_key.items():
        if values:
            out[key] = (min(values), max(values))
    return out


def _dataset_from_json(path: Path) -> SyntheticSDFDataset:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, Mapping):
        raise ValueError(f"cleaning raw_dataset_json must be a mapping: {path}")
    runs_payload = payload.get("runs")
    if not isinstance(runs_payload, list):
        raise ValueError(f"cleaning raw_dataset_json must include list 'runs': {path}")
    runs: list[SyntheticSDFRun] = []
    for index, run in enumerate(runs_payload):
        if not isinstance(run, Mapping):
            raise ValueError(f"cleaning runs[{index}] must be a mapping")
        recipe_raw = run.get("recipe", {})
        if not isinstance(recipe_raw, Mapping):
            raise ValueError(f"cleaning runs[{index}].recipe must be a mapping")
        phi_t_raw = run.get("phi_t")
        if not isinstance(phi_t_raw, list) or not phi_t_raw:
            raise ValueError(f"cleaning runs[{index}].phi_t must be non-empty list")
        phi_t = []
        for frame in phi_t_raw:
            if hasattr(frame, "tolist"):
                frame = frame.tolist()
            if not isinstance(frame, list):
                raise ValueError(f"cleaning runs[{index}] frame must be list")
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


class DataCleaningStage:
    name = "cleaning"

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
        params = runtime.stage_params("cleaning")
        external_inputs = self._stage_external_inputs(runtime)
        input_refs: dict[str, str] = {}

        dataset = runtime.payload.get("dataset_raw")
        if not isinstance(dataset, SyntheticSDFDataset):
            dataset_path = external_inputs.get("raw_dataset_json") or external_inputs.get("dataset_json")
            if dataset_path:
                resolved = Path(dataset_path)
                if not resolved.exists() or not resolved.is_file():
                    raise ValueError(f"cleaning raw_dataset_json does not exist: {resolved}")
                dataset = _dataset_from_json(resolved)
                input_refs["raw_dataset_json"] = str(resolved)
        if not isinstance(dataset, SyntheticSDFDataset):
            raise ValueError("cleaning stage requires dataset_raw or external_inputs.raw_dataset_json")

        remove_duplicates = bool(params.get("remove_duplicates", True))
        fill_missing_recipe = bool(params.get("fill_missing_recipe", True))
        fill_value = float(params.get("fill_missing_recipe_value", 0.0))
        clamp = _parse_bounds_map(params.get("clamp_recipe", {}))
        expected_keys_cfg = params.get("expected_recipe_keys", [])
        expected_keys = [str(key) for key in expected_keys_cfg] if isinstance(expected_keys_cfg, list) else []

        normalize_cfg = params.get("normalize_recipe", False)
        normalize_recipe = False
        normalization_ranges: dict[str, tuple[float, float]] = {}
        if isinstance(normalize_cfg, bool):
            normalize_recipe = normalize_cfg
        elif isinstance(normalize_cfg, dict):
            normalize_recipe = bool(normalize_cfg.get("enabled", True))
            normalization_ranges = _parse_bounds_map(normalize_cfg.get("ranges", {}))

        seen_ids: set[str] = set()
        cleaned_runs = []
        duplicates_removed = 0
        missing_filled = 0
        clamped_values = 0
        normalized_values = 0

        for run in dataset.runs:
            if remove_duplicates and run.run_id in seen_ids:
                duplicates_removed += 1
                continue
            seen_ids.add(run.run_id)

            recipe = {str(key): float(value) for key, value in run.recipe.items()}
            if fill_missing_recipe:
                for key in expected_keys:
                    if key not in recipe:
                        recipe[key] = fill_value
                        missing_filled += 1
            for key in list(recipe.keys()):
                value = recipe[key]
                if value != value and fill_missing_recipe:
                    recipe[key] = fill_value
                    missing_filled += 1

            for key, bounds in clamp.items():
                if key not in recipe:
                    if fill_missing_recipe:
                        recipe[key] = fill_value
                        missing_filled += 1
                    else:
                        continue
                lo, hi = bounds
                clamped = max(lo, min(hi, float(recipe[key])))
                if clamped != recipe[key]:
                    clamped_values += 1
                recipe[key] = clamped

            cleaned_runs.append(replace(run, recipe=recipe))

        normalization_keys: list[str] = []
        if normalize_recipe and cleaned_runs:
            ranges = _resolve_normalization_ranges(runs=cleaned_runs, explicit_ranges=normalization_ranges)
            normalization_keys = sorted(ranges.keys())
            normalized_runs = []
            for run in cleaned_runs:
                recipe = dict(run.recipe)
                for key in normalization_keys:
                    if key not in recipe:
                        if fill_missing_recipe:
                            recipe[key] = fill_value
                            missing_filled += 1
                        else:
                            continue
                    value = float(recipe[key])
                    if value != value:
                        if fill_missing_recipe:
                            value = fill_value
                            missing_filled += 1
                        else:
                            continue
                    lo, hi = ranges[key]
                    span = hi - lo
                    normalized = 0.0 if abs(span) <= 1e-12 else (value - lo) / span
                    if normalized != value:
                        normalized_values += 1
                    recipe[key] = normalized
                normalized_runs.append(replace(run, recipe=recipe))
            cleaned_runs = normalized_runs

        cleaned = SyntheticSDFDataset(runs=cleaned_runs)
        runtime.payload["dataset_clean"] = cleaned

        cleaned_path = write_json(stage_dirs["outputs"] / "cleaned_dataset.json", cleaned.to_dict())
        report = {
            "num_runs_in": len(dataset.runs),
            "num_runs_out": len(cleaned.runs),
            "duplicates_removed": duplicates_removed,
            "missing_recipe_values_filled": missing_filled,
            "clamped_recipe_values": clamped_values,
            "normalized_recipe_values": normalized_values,
            "recipe_normalization_keys": normalization_keys,
            "applied": {
                "remove_duplicates": remove_duplicates,
                "fill_missing_recipe": fill_missing_recipe,
                "normalize_recipe": normalize_recipe,
            },
            "input_refs": input_refs,
        }
        report_path = write_json(stage_dirs["outputs"] / "cleaning_report.json", report)

        return StageResult(
            stage=self.name,
            status="ok",
            metrics={
                "num_runs_in": float(report["num_runs_in"]),
                "num_runs_out": float(report["num_runs_out"]),
                "duplicates_removed": float(duplicates_removed),
                "normalized_recipe_values": float(normalized_values),
            },
            artifacts=[
                ArtifactRef(name="cleaned_dataset", path=str(cleaned_path), kind="json"),
                ArtifactRef(name="cleaning_report", path=str(report_path), kind="json"),
            ],
            details=report,
        )
