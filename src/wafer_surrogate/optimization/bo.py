from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from typing import Any


def _normalize_ranges(ranges: Mapping[str, Sequence[float] | tuple[float, float]]) -> list[tuple[str, float, float]]:
    normalized = [(str(key), min(float(bounds[0]), float(bounds[1])), max(float(bounds[0]), float(bounds[1]))) for key, bounds in ranges.items() if len(bounds) == 2]
    if not normalized:
        raise ValueError("at least one valid search range is required")
    return sorted(normalized, key=lambda item: item[0])


def _vector(candidate: Mapping[str, float], range_items: Sequence[tuple[str, float, float]]) -> tuple[float, ...]:
    vector: list[float] = []
    for key, lo, hi in range_items:
        width = hi - lo
        value = float(candidate[key])
        vector.append(0.5 if width <= 0.0 else (value - lo) / width)
    return tuple(vector)


def _sample(range_items: Sequence[tuple[str, float, float]], rng: random.Random, avoid: set[tuple[int, ...]]) -> dict[str, float]:
    candidate: dict[str, float] = {}
    for _ in range(64):
        candidate = {key: rng.uniform(lo, hi) for key, lo, hi in range_items}
        key = tuple(int(round(v * 1_000_000_000)) for v in _vector(candidate, range_items))
        if key not in avoid:
            return candidate
    return candidate


def _grid_candidate(
    range_items: Sequence[tuple[str, float, float]],
    trial: int,
    num_trials: int,
) -> dict[str, float]:
    if not range_items:
        raise ValueError("at least one search range is required")
    if num_trials <= 1:
        return {key: 0.5 * (lo + hi) for key, lo, hi in range_items}

    dims = len(range_items)
    points_per_axis = max(2, int(math.ceil(float(num_trials) ** (1.0 / float(dims)))))
    axes: list[list[float]] = []
    for _, lo, hi in range_items:
        if hi <= lo:
            axes.append([float(lo)])
            continue
        step = (float(hi) - float(lo)) / float(max(1, points_per_axis - 1))
        axes.append([float(lo) + (step * float(i)) for i in range(points_per_axis)])
    grid = list(product(*axes))
    chosen = grid[int(trial) % len(grid)]
    return {key: float(chosen[idx]) for idx, (key, _, _) in enumerate(range_items)}


def _resolve_mode(strategy: str) -> tuple[str, str, str | None]:
    requested = str(strategy).strip().lower()
    supported = {"random", "grid", "bo", "mfbo"}
    if requested in supported:
        return requested, requested, None
    if requested in {"optuna", "optuna_tpe", "external", "plugin"}:
        return requested, "grid", f"strategy '{requested}' unavailable; fallback to built-in 'grid'"
    return requested, "random", f"strategy '{requested}' unsupported; fallback to built-in 'random'"


def _ei(mean: float, std: float, best: float) -> float:
    if std <= 1e-12:
        return max(0.0, best - mean)
    z = (best - mean) / std
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return max(0.0, ((best - mean) * cdf) + (std * pdf))


def _predict(obs_x: Sequence[tuple[float, ...]], obs_y: Sequence[float], query_x: tuple[float, ...], length_scale: float) -> tuple[float, float]:
    if len(obs_y) <= 1:
        return float(obs_y[0]), 1.0
    denom = 2.0 * max(length_scale * length_scale, 1e-9)
    weights: list[float] = []
    for x in obs_x:
        dist_sq = 0.0
        for xi, qi in zip(x, query_x):
            delta = xi - qi
            dist_sq += delta * delta
        weights.append(math.exp(-dist_sq / denom))
    wsum = sum(weights)
    global_mean = sum(float(y) for y in obs_y) / float(len(obs_y))
    global_var = sum((float(y) - global_mean) ** 2 for y in obs_y) / float(len(obs_y))
    if wsum <= 1e-12:
        return global_mean, max(global_var, 1e-12)
    mean = sum(w * float(y) for w, y in zip(weights, obs_y)) / wsum
    local_var = sum(w * (float(y) - mean) ** 2 for w, y in zip(weights, obs_y)) / wsum
    return mean, max((0.65 * local_var) + (0.35 * global_var), 1e-12)


def run_recipe_search(
    *,
    strategy: str,
    ranges: Mapping[str, Sequence[float] | tuple[float, float]],
    trials: int,
    seed: int,
    evaluate: Callable[[dict[str, float], str], Mapping[str, Any]],
    bo_candidate_pool_size: int = 64,
    mfbo_pool_size: int = 12,
    mfbo_top_k: int = 3,
) -> dict[str, Any]:
    requested_mode, mode, fallback_reason = _resolve_mode(strategy)

    range_items = _normalize_ranges(ranges)
    rng = random.Random(int(seed))
    num_trials = max(1, int(trials))
    init_random_trials = min(4, num_trials)
    pool_size = max(6, int(bo_candidate_pool_size))

    history: list[dict[str, Any]] = []
    obs_x: list[tuple[float, ...]] = []
    obs_y: list[float] = []
    seen_high: set[tuple[int, ...]] = set()
    best_entry: dict[str, Any] | None = None
    low_eval_count = 0

    for trial in range(num_trials):
        low_meta: dict[str, float] = {}
        if mode == "mfbo":
            pool: list[tuple[dict[str, float], float]] = []
            sampled: set[tuple[int, ...]] = set()
            for _ in range(max(3, int(mfbo_pool_size))):
                candidate = _sample(range_items, rng, sampled)
                key = tuple(int(round(v * 1_000_000_000)) for v in _vector(candidate, range_items))
                sampled.add(key)
                low_eval = dict(evaluate(candidate, "low"))
                if "objective" not in low_eval:
                    raise ValueError("evaluation payload must include 'objective'")
                pool.append((candidate, float(low_eval["objective"])))
            pool.sort(key=lambda item: item[1])
            shortlist = [item[0] for item in pool[:min(len(pool), max(1, int(mfbo_top_k)))]]
            low_values = [float(value) for _, value in pool]
            low_meta = {
                "low_pool_best_objective": min(low_values),
                "low_pool_mean_objective": sum(low_values) / float(len(low_values)),
                "low_pool_size": float(len(low_values)),
            }
            low_eval_count += len(low_values)
            if len(obs_y) < 2:
                candidate = shortlist[0]
            else:
                best_seen = min(obs_y)
                length_scale = max(0.08, 0.45 / math.sqrt(float(len(range_items))))
                candidate = max(shortlist, key=lambda cand: _ei(*_predict(obs_x, obs_y, _vector(cand, range_items), length_scale), best_seen))
        elif mode == "grid":
            candidate = _grid_candidate(range_items, trial=trial, num_trials=num_trials)
        elif mode == "bo" and len(obs_y) >= init_random_trials:
            best_seen = min(obs_y)
            length_scale = max(0.08, 0.45 / math.sqrt(float(len(range_items))))
            candidates = [_sample(range_items, rng, seen_high) for _ in range(pool_size)]
            candidate = max(candidates, key=lambda cand: _ei(*_predict(obs_x, obs_y, _vector(cand, range_items), length_scale), best_seen))
        else:
            candidate = _sample(range_items, rng, seen_high)

        high_eval = dict(evaluate(candidate, "high"))
        if "objective" not in high_eval:
            raise ValueError("evaluation payload must include 'objective'")
        objective = float(high_eval["objective"])
        entry: dict[str, Any] = {
            "trial": trial,
            "fidelity": "high",
            "objective": objective,
            "conditions": {key: float(value) for key, value in candidate.items()},
        }
        for key, value in high_eval.items():
            if key != "objective":
                entry[key] = value
        for key, value in low_meta.items():
            entry[key] = float(value)
        history.append(entry)

        candidate_key = tuple(int(round(v * 1_000_000_000)) for v in _vector(candidate, range_items))
        seen_high.add(candidate_key)
        obs_x.append(_vector(candidate, range_items))
        obs_y.append(objective)
        if best_entry is None or objective < float(best_entry["objective"]):
            best_entry = entry

    assert best_entry is not None
    return {
        "requested_strategy": requested_mode,
        "strategy": mode,
        "fallback_reason": fallback_reason,
        "history": history,
        "best_entry": best_entry,
        "fidelity_counts": {"high": len(history), "low": low_eval_count},
    }
