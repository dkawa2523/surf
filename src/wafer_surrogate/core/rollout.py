from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from wafer_surrogate.data.synthetic import SyntheticSDFRun
from wafer_surrogate.geometry import reinit_sussman
from wafer_surrogate.inference.simulate import simulate


def to_float_map(payload: Mapping[str, object]) -> dict[str, float]:
    return {str(key): float(value) for key, value in payload.items()}


def frame_to_list(frame: Any) -> Any:
    if hasattr(frame, "tolist"):
        frame = frame.tolist()
    if not isinstance(frame, list):
        return [[float(frame)]]

    def _convert(payload: Any) -> Any:
        if isinstance(payload, list):
            return [_convert(cell) for cell in payload]
        return float(payload)

    converted = _convert(frame)
    return converted


def fallback_scalar_rollout(
    run: SyntheticSDFRun,
    model: Any,
    *,
    simulation_options: Mapping[str, object] | None = None,
) -> list[list[list[float]]]:
    phi_state = frame_to_list(run.phi_t[0])
    phi_t = [phi_state]
    sim_opts = (
        {str(key): value for key, value in simulation_options.items()}
        if isinstance(simulation_options, Mapping)
        else {}
    )
    reinit_enabled = bool(sim_opts.get("reinit_enabled", False))
    reinit_every_n = max(1, int(sim_opts.get("reinit_every_n", 5)))
    reinit_iters = max(1, int(sim_opts.get("reinit_iters", 8)))
    reinit_dt = float(sim_opts.get("reinit_dt", 0.3))
    reinit_log = sim_opts.get("reinit_log")
    reinit_log_list = reinit_log if isinstance(reinit_log, list) else None
    vn = float(model.predict(run.recipe))
    for step_index in range(1, len(run.phi_t)):
        phi_state = [[float(cell) - float(run.dt) * vn for cell in row] for row in phi_state]
        if reinit_enabled and (step_index % reinit_every_n == 0):
            try:
                phi_state = frame_to_list(
                    reinit_sussman(phi_state, iters=reinit_iters, dt=reinit_dt)
                )
            except Exception:
                pass
            if reinit_log_list is not None:
                reinit_log_list.append(
                    {
                        "step_index": int(step_index),
                        "iters": int(reinit_iters),
                        "dt": float(reinit_dt),
                    }
                )
        phi_t.append(frame_to_list(phi_state))
    return phi_t


def rollout(
    run: SyntheticSDFRun,
    model: Any,
    *,
    simulation_options: Mapping[str, object] | None = None,
) -> list[Any]:
    if hasattr(model, "predict_phi"):
        return [
            model.predict_phi(
                phi0=run.phi_t[0],
                conditions=run.recipe,
                t=float(step) * float(run.dt),
            )
            for step in range(len(run.phi_t))
        ]

    try:
        sim_opts = (
            {str(key): value for key, value in simulation_options.items()}
            if isinstance(simulation_options, Mapping)
            else {}
        )
        return simulate(
            model=model,
            phi0=run.phi_t[0],
            conditions=run.recipe,
            num_steps=len(run.phi_t),
            dt=float(run.dt),
            **sim_opts,
        )
    except Exception:
        return fallback_scalar_rollout(run, model, simulation_options=simulation_options)


def rollout_with_conditions(
    template: SyntheticSDFRun,
    conditions: Mapping[str, object],
    model: Any,
    *,
    simulation_options: Mapping[str, object] | None = None,
) -> list[Any]:
    run = replace(template, recipe=to_float_map(conditions))
    return rollout(run, model, simulation_options=simulation_options)
