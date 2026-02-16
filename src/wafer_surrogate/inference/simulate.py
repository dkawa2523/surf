from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from wafer_surrogate.geometry import levelset_step, reinit_sussman

try:  # Optional dependency in P0.
    import numpy as _np
except Exception:  # pragma: no cover - environment dependent
    _np = None


def _require_numpy() -> Any:
    if _np is None:  # pragma: no cover - exercised only when numpy missing
        raise RuntimeError("numpy is required for wafer_surrogate.inference.simulate")
    return _np


def _to_condition_map(conditions: Mapping[str, float] | None) -> dict[str, float]:
    if not conditions:
        return {}
    return {str(key): float(value) for key, value in conditions.items()}


def _predict_vn_field(
    model: object,
    phi: Any,
    conditions: Mapping[str, float],
    step_index: int,
) -> Any:
    np = _require_numpy()
    if hasattr(model, "predict_vn"):
        vn = model.predict_vn(phi=phi, conditions=conditions, step_index=step_index)
    elif hasattr(model, "predict"):
        vn = model.predict(conditions)  # Backward-compatible scalar prediction.
    else:
        raise TypeError(
            "model contract violation: model must implement predict_vn(phi, conditions, step_index) or predict(features)"
        )

    vn_arr = np.asarray(vn, dtype=float)
    if vn_arr.ndim == 0:
        return np.full_like(phi, float(vn_arr), dtype=float)
    if vn_arr.shape != phi.shape:
        raise ValueError(
            f"predicted vn shape must match phi shape, got {vn_arr.shape} and {phi.shape}"
        )
    return vn_arr


def simulate(
    model: object,
    phi0: Any,
    conditions: Mapping[str, float] | None,
    num_steps: int,
    dt: float,
    spacing: float | Sequence[float] = 1.0,
    reinit_enabled: bool = False,
    reinit_every_n: int = 5,
    reinit_iters: int = 8,
    reinit_dt: float = 0.3,
    reinit_log: list[dict[str, int | float]] | None = None,
) -> list[Any]:
    """Roll out level-set dynamics and return a phi(t) sequence."""
    if num_steps < 2:
        raise ValueError("num_steps must be >= 2")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    np = _require_numpy()
    phi_state = np.asarray(phi0, dtype=float)
    cond_map = _to_condition_map(conditions)
    phi_t = [phi_state.copy()]

    for step_index in range(1, num_steps):
        vn_field = _predict_vn_field(
            model=model,
            phi=phi_state,
            conditions=cond_map,
            step_index=step_index - 1,
        )
        phi_state = levelset_step(phi=phi_state, vn=vn_field, dt=dt, spacing=spacing)
        if bool(reinit_enabled) and int(reinit_every_n) > 0 and (step_index % int(reinit_every_n) == 0):
            phi_state = reinit_sussman(
                phi_state,
                iters=max(1, int(reinit_iters)),
                dt=float(reinit_dt),
                spacing=spacing,
            )
            if reinit_log is not None:
                reinit_log.append(
                    {
                        "step_index": int(step_index),
                        "iters": int(max(1, int(reinit_iters))),
                        "dt": float(reinit_dt),
                    }
                )
        phi_t.append(phi_state.copy())

    return phi_t
