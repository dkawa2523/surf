from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, cast

from wafer_surrogate.data.synthetic import SyntheticSDFDataset


class DatasetSchemaError(ValueError):
    """Schema mismatch for narrow-band SDF dataset."""


class BackendUnavailableError(RuntimeError):
    """Optional backend dependency is not available."""


def _is_seq(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_int(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _to_builtin(value: object) -> object:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if _is_seq(value):
        return [_to_builtin(v) for v in value]
    if _is_int(value):
        return int(value)
    if _is_real(value):
        return float(value)
    return value


def _float_list(name: str, value: object) -> list[float]:
    if not _is_seq(value):
        raise DatasetSchemaError(f"{name} must be a sequence")
    out: list[float] = []
    for idx, item in enumerate(value):
        if not _is_real(item):
            raise DatasetSchemaError(f"{name}[{idx}] must be real-like")
        out.append(float(item))
    return out


def _matrix(
    name: str,
    value: object,
    *,
    kind: str,
    cols: int | None = None,
    enforce_uniform_cols: bool = False,
) -> list[list[int]] | list[list[float]]:
    if not _is_seq(value):
        raise DatasetSchemaError(f"{name} must be a 2D sequence")
    out: list[list[int]] | list[list[float]] = []
    width: int | None = None
    for r_idx, row in enumerate(value):
        if not _is_seq(row):
            raise DatasetSchemaError(f"{name}[{r_idx}] must be a sequence")
        if cols is not None and len(row) != cols:
            raise DatasetSchemaError(f"{name}[{r_idx}] must have {cols} columns")
        if enforce_uniform_cols:
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise DatasetSchemaError(f"{name} rows must share the same width")
        if kind == "int":
            out_row: list[int] = []
            for c_idx, item in enumerate(row):
                if not _is_int(item):
                    raise DatasetSchemaError(f"{name}[{r_idx}][{c_idx}] must be integer-like")
                out_row.append(int(item))
            out.append(out_row)
        else:
            out_row_f: list[float] = []
            for c_idx, item in enumerate(row):
                if not _is_real(item):
                    raise DatasetSchemaError(f"{name}[{r_idx}][{c_idx}] must be real-like")
                out_row_f.append(float(item))
            out.append(out_row_f)
    if enforce_uniform_cols and out and len(out[0]) == 0:
        raise DatasetSchemaError(f"{name} must have at least one feature column")
    return out


@dataclass(frozen=True)
class NarrowBandStep:
    coords: list[list[int]]
    feat: list[list[float]]
    vn_target: list[list[float]]
    priv: list[list[float]] | None = None

    def __post_init__(self) -> None:
        _matrix("steps/*/coords", self.coords, kind="int", cols=3)
        _matrix("steps/*/feat", self.feat, kind="float", enforce_uniform_cols=True)
        _matrix("steps/*/vn_target", self.vn_target, kind="float", cols=1)
        if not (len(self.coords) == len(self.feat) == len(self.vn_target)):
            raise DatasetSchemaError("coords/feat/vn_target row count N must match")
        if self.priv is not None:
            _matrix("steps/*/priv", self.priv, kind="float", enforce_uniform_cols=True)
            if len(self.priv) != len(self.coords):
                raise DatasetSchemaError("priv row count N must match coords")


@dataclass(frozen=True)
class NarrowBandRun:
    run_id: str
    recipe: list[float]
    dt: float
    steps: list[NarrowBandStep]

    def __post_init__(self) -> None:
        if not self.run_id:
            raise DatasetSchemaError("run_id must be non-empty")
        if len(_float_list("meta/recipe", self.recipe)) < 1:
            raise DatasetSchemaError("meta/recipe must have at least one value")
        if not _is_real(self.dt) or float(self.dt) <= 0.0:
            raise DatasetSchemaError("meta/dt must be positive")
        if not self.steps:
            raise DatasetSchemaError("run must have at least one step")


@dataclass(frozen=True)
class NarrowBandDataset:
    runs: list[NarrowBandRun]

    def __post_init__(self) -> None:
        if not self.runs:
            raise DatasetSchemaError("dataset must contain at least one run")
        seen: set[str] = set()
        for run in self.runs:
            if run.run_id in seen:
                raise DatasetSchemaError(f"duplicate run_id: {run.run_id}")
            seen.add(run.run_id)


class InMemoryAdapter:
    def __init__(self) -> None:
        self._root: dict[str, object] = {}

    def _node(self, path: str) -> object:
        if not path:
            return self._root
        node: object = self._root
        for part in (p for p in path.split("/") if p):
            if not isinstance(node, dict) or part not in node:
                raise DatasetSchemaError(f"path not found: {path}")
            node = node[part]
        return node

    def write_array(self, path: str, value: object, *, dtype: str | None = None) -> None:
        del dtype
        parts = [p for p in path.split("/") if p]
        if not parts:
            raise DatasetSchemaError("path must be non-empty")
        node: dict[str, object] = self._root
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise DatasetSchemaError(f"path collision at {part}")
            node = child
        node[parts[-1]] = _to_builtin(value)

    def read_array(self, path: str) -> object:
        node = self._node(path)
        if isinstance(node, dict):
            raise DatasetSchemaError(f"path is group, not dataset: {path}")
        return _to_builtin(node)

    def list_children(self, path: str) -> list[str]:
        node = self._node(path)
        if not isinstance(node, dict):
            raise DatasetSchemaError(f"path is not group: {path}")
        return sorted(node.keys())


class H5pyAdapter:
    def __init__(self, file_obj: object) -> None:
        self._f = file_obj

    def write_array(self, path: str, value: object, *, dtype: str | None = None) -> None:
        parts = [p for p in path.split("/") if p]
        if not parts:
            raise DatasetSchemaError("path must be non-empty")
        group = self._f
        for part in parts[:-1]:
            group = group.require_group(part)
        if parts[-1] in group:
            del group[parts[-1]]
        kwargs = {"data": value}
        if dtype is not None:
            kwargs["dtype"] = dtype
        group.create_dataset(parts[-1], **kwargs)

    def read_array(self, path: str) -> object:
        return _to_builtin(self._f[path][()])

    def list_children(self, path: str) -> list[str]:
        group = self._f[path] if path else self._f
        return sorted(group.keys())


class ZarrAdapter:
    def __init__(self, root_group: object) -> None:
        self._root = root_group

    def write_array(self, path: str, value: object, *, dtype: str | None = None) -> None:
        parts = [p for p in path.split("/") if p]
        if not parts:
            raise DatasetSchemaError("path must be non-empty")
        group = self._root
        for part in parts[:-1]:
            group = group.require_group(part)
        kwargs: dict[str, object] = {"name": parts[-1], "data": value, "overwrite": True}
        if dtype is not None:
            kwargs["dtype"] = dtype
        group.create_dataset(**kwargs)

    def read_array(self, path: str) -> object:
        return _to_builtin(self._root[path][...])

    def list_children(self, path: str) -> list[str]:
        group = self._root[path] if path else self._root
        return sorted(group.keys())


def write_narrow_band_dataset(adapter: object, dataset: NarrowBandDataset) -> None:
    for run in dataset.runs:
        root = f"runs/{run.run_id}"
        adapter.write_array(f"{root}/meta/recipe", run.recipe, dtype="float32")
        adapter.write_array(f"{root}/meta/dt", [run.dt], dtype="float32")
        for k, step in enumerate(run.steps):
            step_root = f"{root}/steps/{k}"
            adapter.write_array(f"{step_root}/coords", step.coords, dtype="int32")
            adapter.write_array(f"{step_root}/feat", step.feat, dtype="float16")
            adapter.write_array(f"{step_root}/vn_target", step.vn_target, dtype="float16")
            if step.priv is not None:
                adapter.write_array(f"{step_root}/priv", step.priv, dtype="float16")


def read_narrow_band_dataset(adapter: object) -> NarrowBandDataset:
    run_ids = adapter.list_children("runs")
    if not run_ids:
        raise DatasetSchemaError("schema requires at least one /runs/{run_id}")
    runs: list[NarrowBandRun] = []
    for run_id in run_ids:
        recipe = _float_list("meta/recipe", adapter.read_array(f"runs/{run_id}/meta/recipe"))
        dt = _float_list("meta/dt", adapter.read_array(f"runs/{run_id}/meta/dt"))
        if len(dt) != 1:
            raise DatasetSchemaError("meta/dt must be shape [1]")
        step_ids = adapter.list_children(f"runs/{run_id}/steps")
        if not step_ids:
            raise DatasetSchemaError("run must contain at least one step")
        ordered = sorted(
            step_ids,
            key=lambda s: (0, int(s)) if s.isdigit() else (1, s),
        )
        steps: list[NarrowBandStep] = []
        for step_id in ordered:
            base = f"runs/{run_id}/steps/{step_id}"
            children = adapter.list_children(base)
            priv = None
            if "priv" in children:
                priv = _matrix("priv", adapter.read_array(f"{base}/priv"), kind="float", enforce_uniform_cols=True)
            steps.append(
                NarrowBandStep(
                    coords=_matrix("coords", adapter.read_array(f"{base}/coords"), kind="int", cols=3),  # type: ignore[arg-type]
                    feat=_matrix("feat", adapter.read_array(f"{base}/feat"), kind="float", enforce_uniform_cols=True),  # type: ignore[arg-type]
                    vn_target=_matrix("vn_target", adapter.read_array(f"{base}/vn_target"), kind="float", cols=1),  # type: ignore[arg-type]
                    priv=priv,  # type: ignore[arg-type]
                )
            )
        runs.append(NarrowBandRun(run_id=run_id, recipe=recipe, dt=dt[0], steps=steps))
    return NarrowBandDataset(runs=runs)


def _load_backend(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover
        raise BackendUnavailableError(f"optional backend '{module_name}' is unavailable") from exc


def write_hdf5_dataset(path: str | Path, dataset: NarrowBandDataset) -> None:
    h5py = _load_backend("h5py")
    with h5py.File(str(path), "w") as fp:
        write_narrow_band_dataset(H5pyAdapter(fp), dataset)


def read_hdf5_dataset(path: str | Path) -> NarrowBandDataset:
    h5py = _load_backend("h5py")
    with h5py.File(str(path), "r") as fp:
        return read_narrow_band_dataset(H5pyAdapter(fp))


def write_zarr_dataset(path: str | Path, dataset: NarrowBandDataset) -> None:
    zarr = _load_backend("zarr")
    write_narrow_band_dataset(ZarrAdapter(zarr.open_group(str(path), mode="w")), dataset)


def read_zarr_dataset(path: str | Path) -> NarrowBandDataset:
    zarr = _load_backend("zarr")
    return read_narrow_band_dataset(ZarrAdapter(zarr.open_group(str(path), mode="r")))


def _rect_grid(name: str, frame: object) -> list[list[float]]:
    grid = cast(
        list[list[float]],
        _matrix(name, frame, kind="float", enforce_uniform_cols=True),
    )
    if not grid:
        raise DatasetSchemaError(f"{name} must contain at least one row")
    if not grid[0]:
        raise DatasetSchemaError(f"{name} must contain at least one column")
    return grid


def _rect_volume(name: str, frame: object) -> list[list[list[float]]]:
    if not _is_seq(frame):
        raise DatasetSchemaError(f"{name} must be a 3D sequence")
    volume: list[list[list[float]]] = []
    height: int | None = None
    width: int | None = None
    for z_idx, plane in enumerate(frame):
        grid = _rect_grid(f"{name}[{z_idx}]", plane)
        if height is None:
            height = len(grid)
            width = len(grid[0])
        elif len(grid) != height or len(grid[0]) != width:
            raise DatasetSchemaError(f"{name} planes must share the same shape")
        volume.append(grid)
    if not volume:
        raise DatasetSchemaError(f"{name} must contain at least one plane")
    return volume


def _grad_norm_2d(frame: list[list[float]]) -> list[list[float]]:
    h = len(frame)
    w = len(frame[0])
    out: list[list[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if w == 1:
                dphi_dx = 0.0
            elif x == 0:
                dphi_dx = frame[y][1] - frame[y][0]
            elif x == w - 1:
                dphi_dx = frame[y][w - 1] - frame[y][w - 2]
            else:
                dphi_dx = 0.5 * (frame[y][x + 1] - frame[y][x - 1])

            if h == 1:
                dphi_dy = 0.0
            elif y == 0:
                dphi_dy = frame[1][x] - frame[0][x]
            elif y == h - 1:
                dphi_dy = frame[h - 1][x] - frame[h - 2][x]
            else:
                dphi_dy = 0.5 * (frame[y + 1][x] - frame[y - 1][x])

            out[y][x] = math.sqrt(dphi_dx * dphi_dx + dphi_dy * dphi_dy)
    return out


def _curvature_proxy_2d(frame: list[list[float]]) -> list[list[float]]:
    h = len(frame)
    w = len(frame[0])
    out: list[list[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            center = frame[y][x]
            left = frame[y][x - 1] if x > 0 else center
            right = frame[y][x + 1] if x + 1 < w else center
            up = frame[y - 1][x] if y > 0 else center
            down = frame[y + 1][x] if y + 1 < h else center
            lap = (left + right + up + down) - (4.0 * center)
            out[y][x] = float(lap)
    return out


def _grad_norm_3d(frame: list[list[list[float]]]) -> list[list[list[float]]]:
    d = len(frame)
    h = len(frame[0])
    w = len(frame[0][0])
    out: list[list[list[float]]] = [[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(d)]
    for z in range(d):
        for y in range(h):
            for x in range(w):
                center = frame[z][y][x]
                if w == 1:
                    dphi_dx = 0.0
                elif x == 0:
                    dphi_dx = frame[z][y][1] - center
                elif x == w - 1:
                    dphi_dx = center - frame[z][y][w - 2]
                else:
                    dphi_dx = 0.5 * (frame[z][y][x + 1] - frame[z][y][x - 1])

                if h == 1:
                    dphi_dy = 0.0
                elif y == 0:
                    dphi_dy = frame[z][1][x] - center
                elif y == h - 1:
                    dphi_dy = center - frame[z][h - 2][x]
                else:
                    dphi_dy = 0.5 * (frame[z][y + 1][x] - frame[z][y - 1][x])

                if d == 1:
                    dphi_dz = 0.0
                elif z == 0:
                    dphi_dz = frame[1][y][x] - center
                elif z == d - 1:
                    dphi_dz = center - frame[d - 2][y][x]
                else:
                    dphi_dz = 0.5 * (frame[z + 1][y][x] - frame[z - 1][y][x])

                out[z][y][x] = math.sqrt((dphi_dx * dphi_dx) + (dphi_dy * dphi_dy) + (dphi_dz * dphi_dz))
    return out


def _curvature_proxy_3d(frame: list[list[list[float]]]) -> list[list[list[float]]]:
    d = len(frame)
    h = len(frame[0])
    w = len(frame[0][0])
    out: list[list[list[float]]] = [[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(d)]
    for z in range(d):
        for y in range(h):
            for x in range(w):
                center = frame[z][y][x]
                left = frame[z][y][x - 1] if x > 0 else center
                right = frame[z][y][x + 1] if x + 1 < w else center
                up = frame[z][y - 1][x] if y > 0 else center
                down = frame[z][y + 1][x] if y + 1 < h else center
                prev_z = frame[z - 1][y][x] if z > 0 else center
                next_z = frame[z + 1][y][x] if z + 1 < d else center
                lap = (left + right + up + down + prev_z + next_z) - (6.0 * center)
                out[z][y][x] = float(lap)
    return out


def synthetic_to_narrow_band_dataset(
    dataset: SyntheticSDFDataset,
    *,
    band_width: float = 0.5,
    min_grad_norm: float = 1e-6,
    include_terminal_step_target: bool = False,
) -> NarrowBandDataset:
    if band_width <= 0:
        raise DatasetSchemaError("band_width must be > 0")
    if min_grad_norm < 0:
        raise DatasetSchemaError("min_grad_norm must be >= 0")
    runs: list[NarrowBandRun] = []
    recipe_keys: list[str] | None = None
    for run in dataset.runs:
        keys = sorted(run.recipe.keys())
        if recipe_keys is None:
            recipe_keys = keys
        if keys != recipe_keys:
            raise DatasetSchemaError("all runs must share recipe keys")
        dt = float(run.dt)
        if dt <= 0:
            raise DatasetSchemaError("run dt must be positive")
        recipe = [float(run.recipe[k]) for k in keys]
        steps: list[NarrowBandStep] = []
        last_step = len(run.phi_t) - 1
        for k, frame in enumerate(run.phi_t):
            if (not include_terminal_step_target) and k >= last_step:
                continue
            next_idx = min(k + 1, len(run.phi_t) - 1)
            coords: list[list[int]] = []
            feat: list[list[float]] = []
            vn_target: list[list[float]] = []

            is_3d = (
                _is_seq(frame)
                and len(frame) > 0
                and _is_seq(frame[0])
                and len(frame[0]) > 0
                and _is_seq(frame[0][0])
            )

            if is_3d:
                frame_grid_3d = _rect_volume(f"phi_t[{k}]", frame)
                nxt_3d = _rect_volume(f"phi_t[{next_idx}]", run.phi_t[next_idx])
                if (
                    len(frame_grid_3d) != len(nxt_3d)
                    or len(frame_grid_3d[0]) != len(nxt_3d[0])
                    or len(frame_grid_3d[0][0]) != len(nxt_3d[0][0])
                ):
                    raise DatasetSchemaError("consecutive phi frames must share the same 3D shape")
                grad_norm_3d = _grad_norm_3d(frame_grid_3d)
                curvature_3d = _curvature_proxy_3d(frame_grid_3d)
                for z, plane in enumerate(frame_grid_3d):
                    for y, row in enumerate(plane):
                        for x, phi in enumerate(row):
                            if abs(phi) > band_width:
                                continue
                            gnorm = grad_norm_3d[z][y][x]
                            if gnorm < min_grad_norm:
                                continue
                            coords.append([x, y, z])
                            feat.append(
                                [
                                    float(phi),
                                    float(gnorm),
                                    float(curvature_3d[z][y][x]),
                                    abs(float(phi)),
                                ]
                            )
                            vn_target.append([(phi - nxt_3d[z][y][x]) / (dt * gnorm)])
            else:
                frame_grid_2d = _rect_grid(f"phi_t[{k}]", frame)
                nxt_2d = _rect_grid(f"phi_t[{next_idx}]", run.phi_t[next_idx])
                if len(frame_grid_2d) != len(nxt_2d) or len(frame_grid_2d[0]) != len(nxt_2d[0]):
                    raise DatasetSchemaError("consecutive phi frames must share the same 2D shape")
                grad_norm_2d = _grad_norm_2d(frame_grid_2d)
                curvature_2d = _curvature_proxy_2d(frame_grid_2d)
                for y, row in enumerate(frame_grid_2d):
                    for x, phi in enumerate(row):
                        if abs(phi) > band_width:
                            continue
                        gnorm = grad_norm_2d[y][x]
                        if gnorm < min_grad_norm:
                            continue
                        coords.append([x, y, 0])
                        feat.append(
                            [
                                float(phi),
                                float(gnorm),
                                float(curvature_2d[y][x]),
                                abs(float(phi)),
                            ]
                        )
                        vn_target.append([(phi - nxt_2d[y][x]) / (dt * gnorm)])
            steps.append(NarrowBandStep(coords=coords, feat=feat, vn_target=vn_target))
        runs.append(NarrowBandRun(run_id=run.run_id, recipe=recipe, dt=dt, steps=steps))
    return NarrowBandDataset(runs=runs)
