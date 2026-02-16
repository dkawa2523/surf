from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from wafer_surrogate.data.synthetic import SyntheticSDFDataset

try:  # Optional dependency.
    import h5py as _h5py
except Exception:  # pragma: no cover - environment dependent
    _h5py = None


PRIV_FEATURE_NAMES = [
    "priv_flux_direct",
    "priv_flux_reflect",
    "priv_angle_hist_0",
    "priv_angle_hist_1",
    "priv_angle_hist_2",
    "priv_energy_hist_0",
    "priv_energy_hist_1",
    "priv_energy_hist_2",
    "priv_reemit_count",
]


def _normalize_hist(values: list[float], bins: int = 3) -> list[float]:
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if not values:
        return [1.0 / float(bins) for _ in range(bins)]
    lo = min(values)
    hi = max(values)
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [1.0 / float(bins) for _ in range(bins)]
    if hi <= lo:
        out = [0.0 for _ in range(bins)]
        out[0] = 1.0
        return out

    width = (hi - lo) / float(bins)
    counts = [0.0 for _ in range(bins)]
    for value in values:
        pos = int((value - lo) / width)
        idx = min(max(pos, 0), bins - 1)
        counts[idx] += 1.0
    denom = max(1.0, float(len(values)))
    return [count / denom for count in counts]


def _flat(frame: Any) -> list[float]:
    out: list[float] = []

    def _walk(value: Any) -> None:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list):
            for cell in value:
                _walk(cell)
            return
        try:
            out.append(float(value))
        except Exception:
            return

    _walk(frame)
    return out


def _proxy_priv_vector(prev_frame: Any, next_frame: Any, dt: float) -> dict[str, float]:
    prev_vals = _flat(prev_frame)
    next_vals = _flat(next_frame)
    if len(prev_vals) != len(next_vals) or not prev_vals or dt <= 0.0:
        raise ValueError("invalid frames for proxy privileged feature generation")

    flux = [(float(p) - float(n)) / float(dt) for p, n in zip(prev_vals, next_vals)]
    abs_flux = [abs(value) for value in flux]
    angle_proxy = []
    for prev, nxt in zip(prev_vals, next_vals):
        delta = float(prev) - float(nxt)
        angle_proxy.append(math.atan2(delta, max(1e-6, abs(float(prev)))))
    energy_proxy = [value * value for value in abs_flux]

    flux_direct = fmean(abs_flux)
    flux_reflect = fmean(abs(float(p) + float(n)) for p, n in zip(prev_vals, next_vals))
    reemit_proxy = pstdev(flux) if len(flux) > 1 else 0.0
    angle_hist = _normalize_hist(angle_proxy, bins=3)
    energy_hist = _normalize_hist(energy_proxy, bins=3)

    out = {
        "priv_flux_direct": float(flux_direct),
        "priv_flux_reflect": float(flux_reflect),
        "priv_angle_hist_0": float(angle_hist[0]),
        "priv_angle_hist_1": float(angle_hist[1]),
        "priv_angle_hist_2": float(angle_hist[2]),
        "priv_energy_hist_0": float(energy_hist[0]),
        "priv_energy_hist_1": float(energy_hist[1]),
        "priv_energy_hist_2": float(energy_hist[2]),
        "priv_reemit_count": float(max(0.0, reemit_proxy)),
    }
    return out


def generate_proxy_privileged_lookup(dataset: SyntheticSDFDataset) -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for run in dataset.runs:
        if len(run.phi_t) < 2:
            continue
        for step_idx in range(len(run.phi_t) - 1):
            out[(str(run.run_id), int(step_idx))] = _proxy_priv_vector(
                run.phi_t[step_idx],
                run.phi_t[step_idx + 1],
                float(run.dt),
            )
    return out


def _vector_from_record(record: Mapping[str, Any]) -> dict[str, float] | None:
    run_id = record.get("run_id")
    step_index = record.get("step_index")
    if run_id is None or step_index is None:
        return None

    try:
        parsed_step = int(step_index)
    except Exception:
        return None

    values: dict[str, float] = {}

    for key in ("priv_flux_direct", "priv_flux_reflect", "priv_reemit_count"):
        if key in record:
            values[key] = float(record[key])

    angle = record.get("priv_angle_hist")
    if isinstance(angle, list) and len(angle) >= 3:
        values["priv_angle_hist_0"] = float(angle[0])
        values["priv_angle_hist_1"] = float(angle[1])
        values["priv_angle_hist_2"] = float(angle[2])

    energy = record.get("priv_energy_hist")
    if isinstance(energy, list) and len(energy) >= 3:
        values["priv_energy_hist_0"] = float(energy[0])
        values["priv_energy_hist_1"] = float(energy[1])
        values["priv_energy_hist_2"] = float(energy[2])

    for key in PRIV_FEATURE_NAMES:
        values.setdefault(key, 0.0)

    values["run_id"] = str(run_id)  # type: ignore[assignment]
    values["step_index"] = float(parsed_step)  # type: ignore[assignment]
    return values


def load_privileged_logs_jsonl(path: str | Path) -> dict[tuple[str, int], dict[str, float]]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise ValueError(f"mc_log_jsonl does not exist: {p}")

    out: dict[tuple[str, int], dict[str, float]] = {}
    with p.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                continue
            row = _vector_from_record(payload)
            if row is None:
                continue
            run_id = str(row.pop("run_id"))
            step = int(float(row.pop("step_index")))
            out[(run_id, step)] = {key: float(row.get(key, 0.0)) for key in PRIV_FEATURE_NAMES}
    return out


def load_privileged_logs_h5(path: str | Path) -> dict[tuple[str, int], dict[str, float]]:
    if _h5py is None:  # pragma: no cover - optional dependency
        raise RuntimeError("h5py is required to read mc_log_h5")

    p = Path(path)
    if not p.exists() or not p.is_file():
        raise ValueError(f"mc_log_h5 does not exist: {p}")

    out: dict[tuple[str, int], dict[str, float]] = {}
    with _h5py.File(str(p), "r") as fp:
        if "runs" not in fp:
            return out
        runs_group = fp["runs"]
        for run_id in runs_group.keys():
            run_group = runs_group[run_id]
            if "steps" not in run_group:
                continue
            steps_group = run_group["steps"]
            for step_key in steps_group.keys():
                step_group = steps_group[step_key]
                try:
                    step_idx = int(step_key)
                except Exception:
                    continue
                vector = {key: 0.0 for key in PRIV_FEATURE_NAMES}
                for key in PRIV_FEATURE_NAMES:
                    if key in step_group:
                        raw = step_group[key][()]
                        if isinstance(raw, (list, tuple)):
                            vector[key] = float(raw[0]) if raw else 0.0
                        else:
                            try:
                                vector[key] = float(raw)
                            except Exception:
                                vector[key] = 0.0
                out[(str(run_id), step_idx)] = vector
    return out


def resolve_privileged_lookup(
    *,
    dataset: SyntheticSDFDataset,
    source: str,
    mc_log_jsonl: str | None,
    mc_log_h5: str | None,
) -> tuple[dict[tuple[str, int], dict[str, float]], str, list[str], dict[str, str]]:
    src = str(source).strip().lower() or "auto"
    warnings: list[str] = []
    input_refs: dict[str, str] = {}

    if src not in {"auto", "real", "proxy"}:
        raise ValueError(f"unsupported priv_source: {source}")

    real_records: dict[tuple[str, int], dict[str, float]] = {}
    if src in {"auto", "real"}:
        if mc_log_jsonl:
            real_records.update(load_privileged_logs_jsonl(mc_log_jsonl))
            input_refs["mc_log_jsonl"] = str(Path(mc_log_jsonl))
        if mc_log_h5:
            real_records.update(load_privileged_logs_h5(mc_log_h5))
            input_refs["mc_log_h5"] = str(Path(mc_log_h5))

    if src == "real":
        if not real_records:
            raise ValueError("priv_source=real requires mc_log_jsonl or mc_log_h5 with valid records")
        return real_records, "real", warnings, input_refs

    if src == "auto" and real_records:
        return real_records, "real", warnings, input_refs

    proxy_records = generate_proxy_privileged_lookup(dataset)
    if src == "auto":
        warnings.append("priv_source auto fallback: real MC logs unavailable, proxy features generated")
    return proxy_records, "proxy", warnings, input_refs


def dense_priv_matrix(vector: Mapping[str, float], n_rows: int) -> list[list[float]]:
    if n_rows < 0:
        raise ValueError("n_rows must be >= 0")
    row = [float(vector.get(name, 0.0)) for name in PRIV_FEATURE_NAMES]
    return [list(row) for _ in range(n_rows)]
