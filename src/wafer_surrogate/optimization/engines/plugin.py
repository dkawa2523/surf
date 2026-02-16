from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any


EvaluateFn = Callable[[dict[str, float], str], Mapping[str, Any]]


class PluginUnavailableError(RuntimeError):
    pass


def run_plugin_engine(
    *,
    plugin_spec: str,
    ranges: Mapping[str, Sequence[float] | tuple[float, float]],
    trials: int,
    seed: int,
    evaluate: EvaluateFn,
) -> dict[str, Any]:
    spec = str(plugin_spec).strip()
    if not spec:
        raise PluginUnavailableError("plugin specification is empty")

    module_name: str
    func_name: str
    if ":" in spec:
        module_name, func_name = spec.split(":", 1)
    else:
        module_name, func_name = spec, "run_plugin_search"

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise PluginUnavailableError(f"plugin module import failed: {module_name}") from exc

    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise PluginUnavailableError(f"plugin callable not found: {module_name}:{func_name}")

    try:
        result = fn(
            ranges=ranges,
            trials=int(trials),
            seed=int(seed),
            evaluate=evaluate,
        )
    except PluginUnavailableError:
        raise
    except Exception as exc:
        raise PluginUnavailableError(f"plugin execution failed: {module_name}:{func_name}") from exc
    if not isinstance(result, Mapping):
        raise PluginUnavailableError("plugin result must be a mapping")

    payload = dict(result)
    payload.setdefault("requested_strategy", "plugin")
    payload.setdefault("strategy", "plugin")
    payload.setdefault("history", [])
    payload.setdefault("best_entry", None)
    payload.setdefault("fidelity_counts", {"high": 0, "low": 0})
    payload["requested_engine"] = f"plugin:{spec}"
    payload["resolved_engine"] = f"plugin:{spec}"
    return payload
