from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any


EvaluateFn = Callable[[dict[str, float], str], Mapping[str, Any]]


def _sample_uniform(
    *,
    keys: list[str],
    bounds: dict[str, tuple[float, float]],
    rng: random.Random,
) -> dict[str, float]:
    return {
        key: float(low + ((high - low) * rng.random()))
        for key, (low, high) in ((key, bounds[key]) for key in keys)
    }


def _require_botorch() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from botorch.acquisition.analytic import ExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("botorch plugin requires optional dependencies: torch, botorch, gpytorch") from exc
    return torch, SingleTaskGP, ExpectedImprovement, fit_gpytorch_mll, optimize_acqf, ExactMarginalLogLikelihood


def run_plugin_search(
    *,
    ranges: Mapping[str, Sequence[float] | tuple[float, float]],
    trials: int,
    seed: int,
    evaluate: EvaluateFn,
) -> dict[str, Any]:
    (
        torch,
        SingleTaskGP,
        ExpectedImprovement,
        fit_gpytorch_mll,
        optimize_acqf,
        ExactMarginalLogLikelihood,
    ) = _require_botorch()

    keys = sorted(str(key) for key in ranges.keys())
    if not keys:
        raise ValueError("botorch plugin requires non-empty ranges")
    bounds = {
        key: (
            float(min(ranges[key][0], ranges[key][1])),  # type: ignore[index]
            float(max(ranges[key][0], ranges[key][1])),  # type: ignore[index]
        )
        for key in keys
    }

    rng = random.Random(int(seed))
    history: list[dict[str, Any]] = []
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    init_trials = max(3, min(8, int(trials) // 4))

    for trial_idx in range(max(1, int(trials))):
        conditions: dict[str, float]
        if trial_idx < init_trials or len(x_rows) < 3:
            conditions = _sample_uniform(keys=keys, bounds=bounds, rng=rng)
        else:
            try:
                train_x = torch.tensor(x_rows, dtype=torch.double)
                train_y = torch.tensor([[-float(v)] for v in y_rows], dtype=torch.double)
                model = SingleTaskGP(train_x, train_y)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
                best_f = float(train_y.max().item())
                acq = ExpectedImprovement(model=model, best_f=best_f, maximize=True)
                acq_bounds = torch.tensor([[bounds[key][0] for key in keys], [bounds[key][1] for key in keys]], dtype=torch.double)
                cand, _ = optimize_acqf(
                    acq_function=acq,
                    bounds=acq_bounds,
                    q=1,
                    num_restarts=4,
                    raw_samples=32,
                )
                candidate = cand.detach().cpu().numpy().reshape(-1).tolist()
                conditions = {key: float(candidate[idx]) for idx, key in enumerate(keys)}
            except Exception:
                conditions = _sample_uniform(keys=keys, bounds=bounds, rng=rng)

        eval_result = dict(evaluate(conditions, "high"))
        objective = float(eval_result.get("objective", 0.0))
        x_rows.append([float(conditions[key]) for key in keys])
        y_rows.append(objective)
        row = {
            "trial": int(trial_idx),
            "fidelity": str(eval_result.get("fidelity", "high")),
            "objective": float(objective),
            "conditions": {str(k): float(v) for k, v in conditions.items()},
            "conditions_used": eval_result.get("conditions_used", {str(k): float(v) for k, v in conditions.items()}),
            "ood_status": eval_result.get("ood_status", "unknown"),
            "ood_distance": eval_result.get("ood_distance"),
            "condition_score": eval_result.get("condition_score"),
            "feature_score": eval_result.get("feature_score"),
            "ood_condition": eval_result.get("ood_condition"),
            "ood_feature": eval_result.get("ood_feature"),
        }
        history.append(row)

    best_entry = min(history, key=lambda item: float(item.get("objective", 0.0)))
    return {
        "requested_strategy": "plugin:botorch",
        "strategy": "plugin:botorch",
        "history": history,
        "best_entry": best_entry,
        "fidelity_counts": {"high": len(history), "low": 0},
        "feature_importance": {},
        "fallback_reason": None,
    }
