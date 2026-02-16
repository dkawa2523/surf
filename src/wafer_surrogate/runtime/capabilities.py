from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache


def _has_module(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _minkowski_engine_status() -> tuple[bool, bool, bool]:
    try:
        module = importlib.import_module("MinkowskiEngine")
    except Exception:
        return False, False, False
    module_file = str(getattr(module, "__file__", "")).replace("\\", "/")
    shim_flag = bool(getattr(module, "__wafer_surrogate_shim__", False))
    real_proxy_flag = bool(getattr(module, "__wafer_surrogate_real_proxy__", False))
    path_flag = "/src/MinkowskiEngine/" in module_file or module_file.endswith("/src/MinkowskiEngine/__init__.py")
    is_shim = bool((shim_flag or path_flag) and not real_proxy_flag)
    return True, not is_shim, is_shim


@dataclass(frozen=True)
class RuntimeCapabilities:
    numpy: bool
    matplotlib: bool
    torch: bool
    minkowski_engine: bool
    minkowski_engine_real: bool
    minkowski_engine_shim: bool
    sbi: bool
    botorch: bool
    optuna: bool

    @property
    def sparse_backend(self) -> bool:
        return bool(self.torch and self.minkowski_engine)

    def missing_summary(self) -> dict[str, bool]:
        return {
            "numpy": bool(self.numpy),
            "matplotlib": bool(self.matplotlib),
            "torch": bool(self.torch),
            "MinkowskiEngine": bool(self.minkowski_engine),
            "MinkowskiEngine_real": bool(self.minkowski_engine_real),
            "MinkowskiEngine_shim": bool(self.minkowski_engine_shim),
            "sbi": bool(self.sbi),
            "botorch": bool(self.botorch),
            "optuna": bool(self.optuna),
            "sparse_backend": bool(self.sparse_backend),
        }


@lru_cache(maxsize=1)
def detect_runtime_capabilities() -> RuntimeCapabilities:
    me_ok, me_real, me_shim = _minkowski_engine_status()
    return RuntimeCapabilities(
        numpy=_has_module("numpy"),
        matplotlib=_has_module("matplotlib"),
        torch=_has_module("torch"),
        minkowski_engine=me_ok,
        minkowski_engine_real=me_real,
        minkowski_engine_shim=me_shim,
        sbi=_has_module("sbi"),
        botorch=_has_module("botorch"),
        optuna=_has_module("optuna"),
    )


def clear_runtime_capabilities_cache() -> None:
    detect_runtime_capabilities.cache_clear()
