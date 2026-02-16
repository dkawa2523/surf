from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from wafer_surrogate.optimization.engines.builtin import run_builtin_engine
from wafer_surrogate.optimization.engines.optuna_engine import OptunaUnavailableError, run_optuna_engine
from wafer_surrogate.optimization.engines.plugin import PluginUnavailableError, run_plugin_engine
from wafer_surrogate.runtime import detect_runtime_capabilities


EvaluateFn = Callable[[dict[str, float], str], Mapping[str, Any]]


SUPPORTED_BUILTIN_STRATEGIES = {"random", "grid", "bo", "mfbo"}


def run_optimization_engine(
    *,
    engine: str,
    strategy: str,
    ranges: Mapping[str, Sequence[float] | tuple[float, float]],
    trials: int,
    seed: int,
    evaluate: EvaluateFn,
    bo_candidate_pool_size: int = 64,
    mfbo_pool_size: int = 12,
    mfbo_top_k: int = 3,
    optuna_sampler: str = "tpe",
) -> dict[str, Any]:
    requested_engine = str(engine).strip().lower()
    requested_strategy = str(strategy).strip().lower()

    if not requested_engine:
        requested_engine = "builtin"

    if requested_engine in SUPPORTED_BUILTIN_STRATEGIES:
        requested_strategy = requested_engine
        requested_engine = "builtin"

    if requested_engine == "builtin":
        strategy_name = requested_strategy if requested_strategy in SUPPORTED_BUILTIN_STRATEGIES else "random"
        result = run_builtin_engine(
            strategy=strategy_name,
            ranges=ranges,
            trials=trials,
            seed=seed,
            evaluate=evaluate,
            bo_candidate_pool_size=bo_candidate_pool_size,
            mfbo_pool_size=mfbo_pool_size,
            mfbo_top_k=mfbo_top_k,
        )
        result.setdefault("fallback_reason", None)
        result["requested_engine"] = "builtin"
        result["resolved_engine"] = "builtin"
        return result

    if requested_engine == "optuna":
        if not detect_runtime_capabilities().optuna:
            fallback = run_builtin_engine(
                strategy="grid",
                ranges=ranges,
                trials=trials,
                seed=seed,
                evaluate=evaluate,
                bo_candidate_pool_size=bo_candidate_pool_size,
                mfbo_pool_size=mfbo_pool_size,
                mfbo_top_k=mfbo_top_k,
            )
            fallback["requested_engine"] = "optuna"
            fallback["resolved_engine"] = "builtin"
            fallback["requested_strategy"] = "optuna"
            fallback["strategy"] = "grid"
            fallback["fallback_reason"] = "engine=optuna requested but optional dependency is unavailable; fallback to builtin:grid"
            return fallback
        try:
            return run_optuna_engine(
                ranges=ranges,
                trials=trials,
                seed=seed,
                evaluate=evaluate,
                sampler=optuna_sampler,
            )
        except OptunaUnavailableError:
            fallback = run_builtin_engine(
                strategy="grid",
                ranges=ranges,
                trials=trials,
                seed=seed,
                evaluate=evaluate,
                bo_candidate_pool_size=bo_candidate_pool_size,
                mfbo_pool_size=mfbo_pool_size,
                mfbo_top_k=mfbo_top_k,
            )
            fallback["requested_engine"] = "optuna"
            fallback["resolved_engine"] = "builtin"
            fallback["requested_strategy"] = "optuna"
            fallback["strategy"] = "grid"
            fallback["fallback_reason"] = "engine=optuna requested but optional dependency is unavailable; fallback to builtin:grid"
            return fallback

    if requested_engine.startswith("plugin:"):
        spec = requested_engine.split(":", 1)[1]
        try:
            return run_plugin_engine(
                plugin_spec=spec,
                ranges=ranges,
                trials=trials,
                seed=seed,
                evaluate=evaluate,
            )
        except PluginUnavailableError:
            fallback = run_builtin_engine(
                strategy="grid",
                ranges=ranges,
                trials=trials,
                seed=seed,
                evaluate=evaluate,
                bo_candidate_pool_size=bo_candidate_pool_size,
                mfbo_pool_size=mfbo_pool_size,
                mfbo_top_k=mfbo_top_k,
            )
            fallback["requested_engine"] = f"plugin:{spec}"
            fallback["resolved_engine"] = "builtin"
            fallback["requested_strategy"] = "plugin"
            fallback["strategy"] = "grid"
            fallback["fallback_reason"] = f"engine=plugin:{spec} requested but plugin is unavailable; fallback to builtin:grid"
            return fallback

    fallback = run_builtin_engine(
        strategy="random",
        ranges=ranges,
        trials=trials,
        seed=seed,
        evaluate=evaluate,
        bo_candidate_pool_size=bo_candidate_pool_size,
        mfbo_pool_size=mfbo_pool_size,
        mfbo_top_k=mfbo_top_k,
    )
    fallback["requested_engine"] = requested_engine
    fallback["resolved_engine"] = "builtin"
    fallback["requested_strategy"] = requested_strategy
    fallback["strategy"] = "random"
    fallback["fallback_reason"] = f"unsupported engine '{requested_engine}'; fallback to builtin:random"
    return fallback


__all__ = ["run_optimization_engine", "SUPPORTED_BUILTIN_STRATEGIES"]
