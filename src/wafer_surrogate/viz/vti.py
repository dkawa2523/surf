from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .utils import is_non_string_sequence as _is_seq

try:
    import numpy as _np
except Exception:  # pragma: no cover - optional dependency
    _np = None


def _format_float(value: float) -> str:
    return f"{float(value):.17g}"


def _to_image_data(field: Any) -> tuple[int, int, int, list[float]]:
    if _np is not None:
        arr = _np.asarray(field, dtype=float)
        if arr.ndim == 2:
            arr = arr.T[:, :, _np.newaxis]
        elif arr.ndim != 3:
            raise ValueError("field must be 2D or 3D")
        nx, ny, nz = [int(dim) for dim in arr.shape]
        flat = [float(value) for value in arr.reshape(-1, order="F")]
        return nx, ny, nz, flat

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
        flat: list[float] = []
        for i, yz_plane in enumerate(field):
            if not _is_seq(yz_plane) or len(yz_plane) != ny:
                raise ValueError(f"inconsistent y-length at x={i}")
            for j, z_line in enumerate(yz_plane):
                if not _is_seq(z_line) or len(z_line) != nz:
                    raise ValueError(f"inconsistent z-length at x={i}, y={j}")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    flat.append(float(field[i][j][k]))
        return nx, ny, nz, flat

    ny = len(field)
    nx = len(first)
    flat = []
    for j, row in enumerate(field):
        if not _is_seq(row) or len(row) != nx:
            raise ValueError(f"inconsistent row length at y={j}")
    for j in range(ny):
        for i in range(nx):
            flat.append(float(field[j][i]))
    return nx, ny, 1, flat


def write_vti_ascii(
    path: str | Path,
    field: Any,
    *,
    array_name: str = "phi",
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nx, ny, nz, flat = _to_image_data(field)
    extent = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
    origin_s = " ".join(_format_float(v) for v in origin)
    spacing_s = " ".join(_format_float(v) for v in spacing)
    lines = [" ".join(_format_float(v) for v in flat[idx:idx + 8]) for idx in range(0, len(flat), 8)]

    with out_path.open("w", encoding="utf-8") as fp:
        fp.write("<?xml version=\"1.0\"?>\n")
        fp.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
        fp.write(f'  <ImageData WholeExtent="{extent}" Origin="{origin_s}" Spacing="{spacing_s}">\n')
        fp.write(f'    <Piece Extent="{extent}">\n')
        fp.write(f'      <PointData Scalars="{array_name}">\n')
        fp.write(f'        <DataArray type="Float64" Name="{array_name}" format="ascii">\n')
        for line in lines:
            fp.write(f"          {line}\n")
        fp.write("        </DataArray>\n")
        fp.write("      </PointData>\n")
        fp.write("      <CellData/>\n")
        fp.write("    </Piece>\n")
        fp.write("  </ImageData>\n")
        fp.write("</VTKFile>\n")
    return out_path


def write_pvd(path: str | Path, datasets: Sequence[tuple[float, str | Path]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fp:
        fp.write("<?xml version=\"1.0\"?>\n")
        fp.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        fp.write("  <Collection>\n")
        for timestep, file_path in datasets:
            candidate = Path(file_path)
            try:
                rel = candidate.resolve().relative_to(out_path.parent.resolve())
                rel_path = rel.as_posix()
            except Exception:
                rel_path = candidate.as_posix()
            fp.write(
                f'    <DataSet timestep="{_format_float(float(timestep))}" group="" part="0" file="{rel_path}"/>\n'
            )
        fp.write("  </Collection>\n")
        fp.write("</VTKFile>\n")
    return out_path


def export_vti_series(
    out_dir: str | Path,
    frames: Sequence[Any],
    *,
    array_name: str = "phi",
    file_prefix: str = "phi",
    times: Sequence[float] | None = None,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if len(frames) == 0:
        raise ValueError("frames must be non-empty")

    if times is None:
        times = [float(step) for step in range(len(frames))]
    if len(times) != len(frames):
        raise ValueError("times length must match frames length")

    vti_paths: list[Path] = []
    datasets: list[tuple[float, Path]] = []
    for index, frame in enumerate(frames):
        frame_path = out_path / f"{file_prefix}_t{index:04d}.vti"
        write_vti_ascii(frame_path, frame, array_name=array_name)
        vti_paths.append(frame_path)
        datasets.append((float(times[index]), frame_path))

    pvd_path = write_pvd(out_path / f"{array_name}.pvd", datasets)
    return {
        "out_dir": out_path,
        "vti_paths": vti_paths,
        "pvd_path": pvd_path,
    }
