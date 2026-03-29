from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wafer_surrogate.data.synthetic import SyntheticSDFDataset, SyntheticSDFRun


def stage_external_inputs(runtime: Any, stage_name: str) -> dict[str, str]:
    run_cfg = getattr(runtime, "run_config", None)
    stages = getattr(run_cfg, "stages", [])
    for stage_cfg in stages:
        if str(getattr(stage_cfg, "name", "")) != str(stage_name):
            continue
        raw = getattr(stage_cfg, "external_inputs", {})
        if isinstance(raw, Mapping):
            return {str(key): str(value) for key, value in raw.items()}
    return {}


def load_json_mapping(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping: {resolved}")
    return {str(key): value for key, value in payload.items()}


def load_float_csv_rows(path: str | Path, *, label: str) -> list[dict[str, float]]:
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8", newline="") as fp:
        rows = [dict(row) for row in csv.DictReader(fp)]
    return [{str(key): float(value) for key, value in row.items()} for row in rows]


def load_float_target_values(path: str | Path, *, label: str, target_key: str = "target") -> list[float]:
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8", newline="") as fp:
        rows = [dict(row) for row in csv.DictReader(fp)]
    if not rows:
        return []
    key = str(target_key)
    if key not in rows[0]:
        candidate_keys = [name for name in rows[0].keys() if name not in {"index", "sample_index"}]
        if not candidate_keys:
            raise ValueError(f"{label} has no usable numeric column: {resolved}")
        key = str(candidate_keys[0])
    return [float(row[key]) for row in rows]


def _to_float_nested(payload: Any) -> Any:
    if hasattr(payload, "tolist"):
        payload = payload.tolist()
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [_to_float_nested(cell) for cell in payload]
    return float(payload)


def dataset_from_mapping(payload: Mapping[str, Any], *, label: str) -> SyntheticSDFDataset:
    runs_payload = payload.get("runs")
    if not isinstance(runs_payload, list) or not runs_payload:
        raise ValueError(f"{label} must include non-empty 'runs'")
    runs: list[SyntheticSDFRun] = []
    for index, run in enumerate(runs_payload):
        if not isinstance(run, Mapping):
            raise ValueError(f"{label} runs[{index}] must be a mapping")
        recipe_raw = run.get("recipe", {})
        if not isinstance(recipe_raw, Mapping):
            raise ValueError(f"{label} runs[{index}].recipe must be a mapping")
        phi_t_raw = run.get("phi_t")
        if not isinstance(phi_t_raw, list) or not phi_t_raw:
            raise ValueError(f"{label} runs[{index}].phi_t must be non-empty list")
        phi_t = [_to_float_nested(frame) for frame in phi_t_raw]
        runs.append(
            SyntheticSDFRun(
                run_id=str(run.get("run_id", f"run_{index:03d}")),
                dt=float(run.get("dt", 0.1)),
                recipe={str(key): float(value) for key, value in recipe_raw.items()},
                phi_t=phi_t,
            )
        )
    return SyntheticSDFDataset(runs=runs)


def dataset_from_json(path: str | Path, *, label: str) -> SyntheticSDFDataset:
    payload = load_json_mapping(path, label=label)
    return dataset_from_mapping(payload, label=label)


def normalize_artifact_path(path: str | Path) -> str:
    return Path(path).as_posix()
