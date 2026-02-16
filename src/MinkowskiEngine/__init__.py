from __future__ import annotations

from dataclasses import dataclass
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn


def _shim_src_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _search_paths_without_shim() -> list[str]:
    src_root = _shim_src_root()
    out: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except Exception:
            out.append(entry)
            continue
        if resolved == src_root:
            continue
        out.append(entry)
    return out


def _load_real_module_if_requested() -> bool:
    require_real = os.environ.get("WAFER_SURROGATE_REQUIRE_REAL_ME", "").strip().lower() in {"1", "true", "yes", "on"}
    if not require_real:
        return False

    spec = importlib.machinery.PathFinder.find_spec("MinkowskiEngine", _search_paths_without_shim())
    if spec is None or spec.loader is None or spec.origin is None:
        raise ImportError("WAFER_SURROGATE_REQUIRE_REAL_ME=1 but real MinkowskiEngine is not installed")
    origin = str(spec.origin).replace("\\", "/")
    if origin.endswith("/src/MinkowskiEngine/__init__.py"):
        raise ImportError("WAFER_SURROGATE_REQUIRE_REAL_ME=1 but only local shim was found")

    module = importlib.util.module_from_spec(spec)
    sys.modules["MinkowskiEngine"] = module
    spec.loader.exec_module(module)
    globals().update(module.__dict__)
    globals()["__wafer_surrogate_shim__"] = False
    globals()["__wafer_surrogate_real_proxy__"] = True
    globals()["__wafer_surrogate_real_module_file__"] = str(getattr(module, "__file__", ""))
    return True


if _load_real_module_if_requested():
    pass
else:
    __wafer_surrogate_shim__ = True


    @dataclass
    class _CoordinateHandle:
        coordinates: torch.Tensor


    class SparseTensor:
        """Lightweight compatibility shim for MinkowskiEngine SparseTensor."""

        def __init__(
            self,
            features: torch.Tensor,
            coordinates: torch.Tensor | None = None,
            *,
            coordinate_map_key: Any | None = None,
            coordinate_manager: Any | None = None,
        ) -> None:
            if not isinstance(features, torch.Tensor):
                raise TypeError("SparseTensor.features must be torch.Tensor")
            self.F = features
            self.coordinate_map_key = coordinate_map_key
            self.coordinate_manager = coordinate_manager
            if coordinates is not None:
                self.C = coordinates
                if self.coordinate_manager is None:
                    self.coordinate_manager = _CoordinateHandle(coordinates=coordinates)
            else:
                handle = self.coordinate_manager
                if isinstance(handle, _CoordinateHandle):
                    self.C = handle.coordinates
                elif hasattr(handle, "coordinates"):
                    self.C = handle.coordinates  # type: ignore[assignment]
                else:
                    # Fallback for operations that don't depend on coordinates.
                    self.C = torch.zeros((self.F.shape[0], 4), dtype=torch.int32, device=self.F.device)


    class MinkowskiConvolution(nn.Module):
        def __init__(
            self,
            *,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,  # noqa: ARG002
            stride: int = 1,  # noqa: ARG002
            dimension: int = 3,  # noqa: ARG002
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(int(in_channels), int(out_channels))

        def forward(self, x: SparseTensor) -> SparseTensor:
            y = self.linear(x.F)
            return SparseTensor(
                y,
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )


    class MinkowskiBatchNorm(nn.Module):
        def __init__(self, num_features: int) -> None:
            super().__init__()
            self.bn = nn.BatchNorm1d(int(num_features))

        def forward(self, x: SparseTensor) -> SparseTensor:
            y = self.bn(x.F)
            return SparseTensor(
                y,
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )


    class MinkowskiReLU(nn.Module):
        def forward(self, x: SparseTensor) -> SparseTensor:
            y = torch.relu(x.F)
            return SparseTensor(
                y,
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )
