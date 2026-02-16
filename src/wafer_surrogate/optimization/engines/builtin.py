from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from wafer_surrogate.optimization.bo import run_recipe_search


EvaluateFn = Callable[[dict[str, float], str], Mapping[str, Any]]


def run_builtin_engine(
    *,
    strategy: str,
    ranges: Mapping[str, Sequence[float] | tuple[float, float]],
    trials: int,
    seed: int,
    evaluate: EvaluateFn,
    bo_candidate_pool_size: int = 64,
    mfbo_pool_size: int = 12,
    mfbo_top_k: int = 3,
) -> dict[str, Any]:
    result = run_recipe_search(
        strategy=strategy,
        ranges=ranges,
        trials=trials,
        seed=seed,
        evaluate=evaluate,
        bo_candidate_pool_size=bo_candidate_pool_size,
        mfbo_pool_size=mfbo_pool_size,
        mfbo_top_k=mfbo_top_k,
    )
    result["requested_engine"] = "builtin"
    result["resolved_engine"] = "builtin"
    return result
