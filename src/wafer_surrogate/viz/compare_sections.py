from __future__ import annotations

import json
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
            # 2D (z, x) -> (x, y=1, z) for XZ compare
            arr = arr.T[:, _np.newaxis, :]
        elif int(arr.ndim) != 3:
            raise ValueError("field must be 2D or 3D")
        nx, ny, nz = [int(dim) for dim in arr.shape]
        return [
            [[float(arr[x, y, z]) for z in range(nz)] for y in range(ny)]
            for x in range(nx)
        ]

    if not _is_seq(field) or len(field) == 0 or not _is_seq(field[0]) or len(field[0]) == 0:
        raise ValueError("field must be a non-empty 2D or 3D sequence")

    first_first = field[0][0]
    if _is_seq(first_first):
        nx = len(field)
        ny = len(field[0])
        nz = len(first_first)
        return [
            [
                [float(field[x][y][z]) for z in range(nz)]
                for y in range(ny)
            ]
            for x in range(nx)
        ]

    nz = len(field)
    nx = len(field[0])
    grid = [[[0.0 for _ in range(nz)]] for _ in range(nx)]
    for z in range(nz):
        if not _is_seq(field[z]) or len(field[z]) != nx:
            raise ValueError(f"inconsistent row length at z={z}")
        for x in range(nx):
            grid[x][0][z] = float(field[z][x])
    return grid


def _resolve_index(requested: int | None, size: int) -> int:
    if size <= 0:
        raise ValueError("axis size must be > 0")
    if requested is None:
        return size // 2
    if requested < 0 or requested >= size:
        raise ValueError(f"index {requested} is out of range for size={size}")
    return int(requested)


def _extract_xz_plane(
    grid: list[list[list[float]]],
    *,
    y_index: int | None,
) -> tuple[dict[str, int], list[list[float]]]:
    nx = len(grid)
    ny = len(grid[0])
    nz = len(grid[0][0])
    y_idx = _resolve_index(y_index, ny)
    plane = [[float(grid[x][y_idx][z]) for x in range(nx)] for z in range(nz)]
    return {"nx": nx, "ny": ny, "nz": nz, "y": y_idx}, plane


def _iso_points(plane: Sequence[Sequence[float]], contour_level: float) -> list[list[float]]:
    height = len(plane)
    width = len(plane[0]) if height else 0
    points: list[list[float]] = []
    if width < 2 or height < 2:
        return points

    for z in range(height):
        for x in range(width - 1):
            a = float(plane[z][x]) - float(contour_level)
            b = float(plane[z][x + 1]) - float(contour_level)
            if a == 0.0:
                points.append([float(x), float(z)])
            if (a < 0.0 and b > 0.0) or (a > 0.0 and b < 0.0):
                t = abs(a) / (abs(a) + abs(b))
                points.append([float(x) + float(t), float(z)])

    for z in range(height - 1):
        for x in range(width):
            a = float(plane[z][x]) - float(contour_level)
            b = float(plane[z + 1][x]) - float(contour_level)
            if (a < 0.0 and b > 0.0) or (a > 0.0 and b < 0.0):
                t = abs(a) / (abs(a) + abs(b))
                points.append([float(x), float(z) + float(t)])

    return points


def _render_png(
    out_path: Path,
    pred_plane: Sequence[Sequence[float]],
    gt_plane: Sequence[Sequence[float]],
    *,
    contour_level: float,
    split: str,
    sample_index: int,
    y_index: int,
    frame_index: int | None,
    plt: Any,
) -> None:
    from matplotlib.lines import Line2D

    fig, axis = plt.subplots(1, 1, figsize=(6.2, 4.8))
    height = len(pred_plane)
    width = len(pred_plane[0]) if height else 0
    if width > 1 and height > 1:
        axis.contour(range(width), range(height), gt_plane, levels=[float(contour_level)], colors="#1f77b4", linewidths=1.3, linestyles="--")
        axis.contour(range(width), range(height), pred_plane, levels=[float(contour_level)], colors="#d62728", linewidths=1.3, linestyles="-")
    title = f"{split} sample={sample_index:03d}, y={y_index:03d}"
    if frame_index is not None:
        title = f"{title}, t={frame_index:04d}"
    axis.set_title(title)
    axis.set_xlabel("x index")
    axis.set_ylabel("z index")
    axis.legend(
        [
            Line2D([0], [0], color="#d62728", lw=1.3, linestyle="-"),
            Line2D([0], [0], color="#1f77b4", lw=1.3, linestyle="--"),
        ],
        ["pred φ=0", "gt φ=0"],
        loc="best",
    )
    axis.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_compare_section(
    *,
    pred_field: Any,
    gt_field: Any,
    out_dir: str | Path,
    split: str = "val",
    sample_index: int = 0,
    y_index: int | None = None,
    contour_level: float = 0.0,
    frame_index: int | None = None,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pred_shape, pred_plane = _extract_xz_plane(_to_xyz_grid(pred_field), y_index=y_index)
    gt_shape, gt_plane = _extract_xz_plane(_to_xyz_grid(gt_field), y_index=pred_shape["y"])
    if (pred_shape["nx"], pred_shape["ny"], pred_shape["nz"]) != (gt_shape["nx"], gt_shape["ny"], gt_shape["nz"]):
        raise ValueError("pred and gt shapes must match for compare-sections")

    safe_split = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in split) or "split"
    stem = f"{safe_split}_sample{int(sample_index):03d}_y{int(pred_shape['y']):03d}"
    if frame_index is not None:
        stem += f"_t{int(frame_index):04d}"

    plt = _load_pyplot()
    png_path: Path | None = None
    fallback_path: Path | None = None
    if plt is not None:
        png_path = out_path / f"{stem}.png"
        _render_png(
            out_path=png_path,
            pred_plane=pred_plane,
            gt_plane=gt_plane,
            contour_level=float(contour_level),
            split=safe_split,
            sample_index=int(sample_index),
            y_index=int(pred_shape["y"]),
            frame_index=frame_index,
            plt=plt,
        )
    else:
        fallback_path = out_path / f"{stem}.contours.json"
        with fallback_path.open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "split": safe_split,
                    "sample_index": int(sample_index),
                    "frame_index": None if frame_index is None else int(frame_index),
                    "shape": {
                        "nx": int(pred_shape["nx"]),
                        "ny": int(pred_shape["ny"]),
                        "nz": int(pred_shape["nz"]),
                    },
                    "y_index": int(pred_shape["y"]),
                    "contour_level": float(contour_level),
                    "pred_points": _iso_points(pred_plane, contour_level=float(contour_level)),
                    "gt_points": _iso_points(gt_plane, contour_level=float(contour_level)),
                    "note": "matplotlib unavailable; contour crossing points are written as fallback.",
                },
                fp,
                indent=2,
            )

    return {
        "out_dir": out_path,
        "shape": {"nx": int(pred_shape["nx"]), "ny": int(pred_shape["ny"]), "nz": int(pred_shape["nz"])},
        "y_index": int(pred_shape["y"]),
        "contour_level": float(contour_level),
        "matplotlib_available": plt is not None,
        "png_path": png_path,
        "fallback_path": fallback_path,
    }
