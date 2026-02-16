from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


EvaluateFn = Callable[[dict[str, float], str], Mapping[str, Any]]


class OptunaUnavailableError(RuntimeError):
    pass


def run_optuna_engine(
    *,
    ranges: Mapping[str, Sequence[float] | tuple[float, float]],
    trials: int,
    seed: int,
    evaluate: EvaluateFn,
    sampler: str = "tpe",
) -> dict[str, Any]:
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - optional dependency
        raise OptunaUnavailableError("optuna is not installed") from exc

    normalized: dict[str, tuple[float, float]] = {}
    for key, bounds in ranges.items():
        if len(bounds) != 2:
            continue
        lo = float(min(bounds[0], bounds[1]))
        hi = float(max(bounds[0], bounds[1]))
        normalized[str(key)] = (lo, hi)
    if not normalized:
        raise ValueError("at least one valid range is required")

    sampler_name = str(sampler).strip().lower()
    if sampler_name == "random":
        study_sampler: Any = optuna.samplers.RandomSampler(seed=int(seed))
    else:
        study_sampler = optuna.samplers.TPESampler(seed=int(seed))

    history: list[dict[str, Any]] = []

    def _objective(trial: Any) -> float:
        candidate: dict[str, float] = {}
        for key, (lo, hi) in normalized.items():
            candidate[key] = float(trial.suggest_float(key, lo, hi))
        result = dict(evaluate(candidate, "high"))
        objective = float(result.get("objective"))
        history.append(
            {
                "trial": int(trial.number),
                "fidelity": "high",
                "objective": objective,
                "conditions": candidate,
                **{k: v for k, v in result.items() if k != "objective"},
            }
        )
        return objective

    study = optuna.create_study(direction="minimize", sampler=study_sampler)
    study.optimize(_objective, n_trials=max(1, int(trials)))

    best = min(history, key=lambda row: float(row["objective"]))
    feature_importance: dict[str, float] = {}
    try:
        raw_importance = optuna.importance.get_param_importances(study)
        if isinstance(raw_importance, dict):
            feature_importance = {str(k): float(v) for k, v in raw_importance.items()}
    except Exception:
        feature_importance = {}
    return {
        "requested_strategy": "optuna",
        "strategy": "optuna",
        "fallback_reason": None,
        "history": history,
        "best_entry": best,
        "fidelity_counts": {"high": len(history), "low": 0},
        "requested_engine": "optuna",
        "resolved_engine": "optuna",
        "feature_importance": feature_importance,
    }
