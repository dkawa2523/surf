from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .utils import is_non_string_sequence as _is_seq, load_pyplot as _load_pyplot

try:
    import numpy as _np
except Exception:  # pragma: no cover - optional dependency
    _np = None


def _to_xyz_grid(field: Any) -> list[list[list[float]]]:
    if _np is not None:
        arr = _np.asarray(field, dtype=float)
        if int(arr.ndim) == 2:
            arr = arr.T[:, :, _np.newaxis]
        elif int(arr.ndim) != 3:
            raise ValueError("field must be 2D or 3D")
        nx, ny, nz = [int(dim) for dim in arr.shape]
        return [
            [[float(arr[i, j, k]) for k in range(nz)] for j in range(ny)]
            for i in range(nx)
        ]

    if not _is_seq(field) or len(field) == 0:
        raise ValueError("field must be a non-empty 2D or 3D sequence")
    first = field[0]
    if not _is_seq(first) or len(first) == 0:
        raise ValueError("field must be a non-empty 2D or 3D sequence")

    first_first = first[0]
    if _is_seq(first_first):
        nx = len(field)
        ny = len(first)
        nz = len(first_first)
        grid: list[list[list[float]]] = []
        for x_idx, yz_plane in enumerate(field):
            if not _is_seq(yz_plane) or len(yz_plane) != ny:
                raise ValueError(f"inconsistent y-length at x={x_idx}")
            y_rows: list[list[float]] = []
            for y_idx, z_line in enumerate(yz_plane):
                if not _is_seq(z_line) or len(z_line) != nz:
                    raise ValueError(f"inconsistent z-length at x={x_idx}, y={y_idx}")
                y_rows.append([float(value) for value in z_line])
            grid.append(y_rows)
        return grid

    ny = len(field)
    nx = len(first)
    grid = [[[0.0] for _ in range(ny)] for _ in range(nx)]
    for y_idx, row in enumerate(field):
        if not _is_seq(row) or len(row) != nx:
            raise ValueError(f"inconsistent row length at y={y_idx}")
        for x_idx in range(nx):
            grid[x_idx][y_idx][0] = float(row[x_idx])
    return grid


def _resolve_index(requested: int | None, size: int, axis_name: str) -> int:
    if size <= 0:
        raise ValueError(f"{axis_name} axis must be non-empty")
    if requested is None:
        return size // 2
    if requested < 0 or requested >= size:
        raise ValueError(f"{axis_name}-index {requested} is out of range for size={size}")
    return int(requested)


def _slice_planes(
    grid: list[list[list[float]]],
    *,
    x_index: int | None,
    y_index: int | None,
    z_index: int | None,
) -> tuple[dict[str, int], dict[str, list[list[float]]]]:
    nx = len(grid)
    ny = len(grid[0])
    nz = len(grid[0][0])
    x_idx = _resolve_index(x_index, nx, "x")
    y_idx = _resolve_index(y_index, ny, "y")
    z_idx = _resolve_index(z_index, nz, "z")

    xy = [[float(grid[x][y][z_idx]) for x in range(nx)] for y in range(ny)]
    xz = [[float(grid[x][y_idx][z]) for x in range(nx)] for z in range(nz)]
    yz = [[float(grid[x_idx][y][z]) for y in range(ny)] for z in range(nz)]
    return {"x": x_idx, "y": y_idx, "z": z_idx}, {"xy": xy, "xz": xz, "yz": yz}


def _render_png(
    out_path: Path,
    planes: dict[str, list[list[float]]],
    indices: dict[str, int],
    contour_level: float,
    plt: Any,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    plane_specs = [
        ("XY", planes["xy"], f"z={indices['z']}"),
        ("XZ", planes["xz"], f"y={indices['y']}"),
        ("YZ", planes["yz"], f"x={indices['x']}"),
    ]
    for axis, (name, plane, fixed_axis) in zip(axes, plane_specs):
        image = axis.imshow(plane, origin="lower", cmap="coolwarm")
        height = len(plane)
        width = len(plane[0]) if height else 0
        if width > 1 and height > 1:
            axis.contour(
                range(width),
                range(height),
                plane,
                levels=[float(contour_level)],
                colors="black",
                linewidths=0.8,
            )
        axis.set_title(f"{name} ({fixed_axis})")
        axis.set_xlabel("index")
        axis.set_ylabel("index")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_slice_quicklook_series(
    frames: Sequence[Any],
    out_dir: str | Path,
    *,
    x_index: int | None = None,
    y_index: int | None = None,
    z_index: int | None = None,
    contour_level: float = 0.0,
    file_prefix: str = "slices",
) -> dict[str, Any]:
    if not frames:
        raise ValueError("frames must be non-empty")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    plt = _load_pyplot()
    has_matplotlib = plt is not None

    png_paths: list[Path] = []
    fallback_paths: list[Path] = []
    last_shape: dict[str, int] = {}
    resolved_indices: dict[str, int] = {}
    for frame_idx, frame in enumerate(frames):
        grid = _to_xyz_grid(frame)
        indices, planes = _slice_planes(
            grid,
            x_index=x_index,
            y_index=y_index,
            z_index=z_index,
        )
        nx = len(grid)
        ny = len(grid[0])
        nz = len(grid[0][0])
        last_shape = {"nx": nx, "ny": ny, "nz": nz}
        resolved_indices = indices
        if has_matplotlib:
            png_path = out_path / f"{file_prefix}_t{frame_idx:04d}.png"
            _render_png(png_path, planes, indices, contour_level, plt)
            png_paths.append(png_path)
            continue
        fallback_path = out_path / f"{file_prefix}_t{frame_idx:04d}.fallback.json"
        with fallback_path.open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "frame_index": frame_idx,
                    "shape": {"nx": nx, "ny": ny, "nz": nz},
                    "slice_indices": indices,
                    "slices": planes,
                    "note": "matplotlib unavailable; numeric slices are written as fallback.",
                },
                fp,
                indent=2,
            )
        fallback_paths.append(fallback_path)

    return {
        "out_dir": out_path,
        "png_paths": png_paths,
        "fallback_paths": fallback_paths,
        "matplotlib_available": has_matplotlib,
        "shape": last_shape,
        "slice_indices": resolved_indices,
    }


def _read_vti_ascii(path: Path, *, array_name: str = "phi") -> list[list[list[float]]]:
    tree = ET.parse(path)
    root = tree.getroot()
    image = root.find(".//ImageData")
    if image is None:
        raise ValueError(f"ImageData section is missing: {path}")
    extent_raw = image.attrib.get("WholeExtent")
    if extent_raw is None:
        raise ValueError(f"WholeExtent is missing: {path}")
    ext = [int(token) for token in extent_raw.split()]
    if len(ext) != 6:
        raise ValueError(f"WholeExtent must have 6 integers: {path}")
    nx = (ext[1] - ext[0]) + 1
    ny = (ext[3] - ext[2]) + 1
    nz = (ext[5] - ext[4]) + 1
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"invalid extent in {path}")

    data_array = root.find(f".//DataArray[@Name='{array_name}']")
    if data_array is None:
        data_array = root.find(".//DataArray")
    if data_array is None:
        raise ValueError(f"DataArray is missing: {path}")
    text = data_array.text or ""
    values = [float(token) for token in text.split()]
    expected = nx * ny * nz
    if len(values) != expected:
        raise ValueError(f"DataArray length mismatch in {path}: expected {expected}, got {len(values)}")

    grid = [[[0.0 for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]
    index = 0
    for z_idx in range(nz):
        for y_idx in range(ny):
            for x_idx in range(nx):
                grid[x_idx][y_idx][z_idx] = float(values[index])
                index += 1
    return grid


def load_vti_series(
    vti_dir: str | Path,
    *,
    pattern: str = "phi_t*.vti",
    array_name: str = "phi",
) -> list[list[list[list[float]]]]:
    directory = Path(vti_dir)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"vti-dir does not exist: {directory}")
    paths = sorted(directory.glob(pattern))
    if not paths:
        paths = sorted(directory.glob("*.vti"))
    if not paths:
        raise ValueError(f"vti-dir does not contain .vti files: {directory}")
    return [_read_vti_ascii(path, array_name=array_name) for path in paths]
