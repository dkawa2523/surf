#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import struct
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from wafer_surrogate.data.io import (
    NarrowBandDataset,
    NarrowBandRun,
    NarrowBandStep,
    write_hdf5_dataset,
)
from wafer_surrogate.data.synthetic import SyntheticSDFDataset, SyntheticSDFRun

try:
    from scipy.ndimage import distance_transform_edt as _distance_transform_edt
except Exception:  # pragma: no cover - optional dependency
    _distance_transform_edt = None


def _parse_run_list(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _load_index_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = [dict(row) for row in csv.DictReader(fp)]
    if not rows:
        raise ValueError(f"index.csv has no rows: {path}")
    return rows


def _to_float(value: str) -> float | None:
    raw = str(value).strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _numeric_recipe_from_index(row: dict[str, str]) -> dict[str, float]:
    skip = {"run_id", "run_dir", "status", "error"}
    out: dict[str, float] = {}
    for key, value in row.items():
        if key in skip:
            continue
        parsed = _to_float(value)
        if parsed is not None:
            out[str(key)] = float(parsed)
    return out


def _numeric_recipe_from_inputs(path: Path) -> tuple[dict[str, float], float]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"inputs.json must be a JSON object: {path}")
    recipe: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            recipe[str(key)] = float(value)
    seg = payload.get("segment_duration_s")
    total = payload.get("total_duration_s")
    snaps = payload.get("n_snapshots")
    if isinstance(seg, (int, float)) and float(seg) > 0.0:
        return recipe, float(seg)
    if isinstance(total, (int, float)) and isinstance(snaps, (int, float)) and float(snaps) > 0:
        return recipe, float(total) / float(snaps)
    return recipe, 1.0


def _extents_to_point_shape(image_data: ET.Element) -> tuple[int, int, int]:
    raw_extent = image_data.attrib.get("WholeExtent")
    if raw_extent is None:
        raise ValueError("ImageData/WholeExtent is missing")
    ext = [int(token) for token in raw_extent.split()]
    if len(ext) != 6:
        raise ValueError(f"invalid WholeExtent: {raw_extent}")
    nx = (ext[1] - ext[0]) + 1
    ny = (ext[3] - ext[2]) + 1
    nz = (ext[5] - ext[4]) + 1
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"invalid extent values: {raw_extent}")
    return nx, ny, nz


def _extents_to_cell_shape(image_data: ET.Element) -> tuple[int, int, int]:
    nx, ny, nz = _extents_to_point_shape(image_data)
    return max(nx - 1, 1), max(ny - 1, 1), max(nz - 1, 1)


def _dtype_from_vtk(vtk_type: str) -> np.dtype[Any]:
    mapping = {
        "Int8": np.dtype("<i1"),
        "UInt8": np.dtype("<u1"),
        "Int16": np.dtype("<i2"),
        "UInt16": np.dtype("<u2"),
        "Int32": np.dtype("<i4"),
        "UInt32": np.dtype("<u4"),
        "Float32": np.dtype("<f4"),
        "Float64": np.dtype("<f8"),
    }
    out = mapping.get(vtk_type)
    if out is None:
        raise ValueError(f"unsupported VTK DataArray type: {vtk_type}")
    return out


def _uint_unpack_format(header_type: str) -> tuple[str, int]:
    if header_type == "UInt32":
        return "<I", 4
    if header_type == "UInt64":
        return "<Q", 8
    raise ValueError(f"unsupported VTK header_type: {header_type}")


def _decode_appended_array(
    *,
    encoded_payload: str,
    offset: int,
    next_offset: int,
    header_type: str,
    compressor: str,
) -> bytes:
    if offset < 0 or next_offset <= offset:
        raise ValueError(f"invalid array offsets: offset={offset}, next_offset={next_offset}")
    segment = encoded_payload[offset:next_offset]
    if not segment:
        raise ValueError(f"empty appended segment: offset={offset}, next_offset={next_offset}")
    raw = base64.b64decode(segment)
    if not compressor:
        return raw

    unpack_fmt, uint_size = _uint_unpack_format(header_type)
    cursor = 0
    if len(raw) < (3 * uint_size):
        raise ValueError("compressed segment header is too short")

    num_blocks = int(struct.unpack_from(unpack_fmt, raw, cursor)[0])
    cursor += uint_size
    _ = int(struct.unpack_from(unpack_fmt, raw, cursor)[0])
    cursor += uint_size
    _ = int(struct.unpack_from(unpack_fmt, raw, cursor)[0])
    cursor += uint_size

    compressed_sizes: list[int] = []
    for _idx in range(num_blocks):
        if cursor + uint_size > len(raw):
            raise ValueError("compressed header is truncated")
        compressed_sizes.append(int(struct.unpack_from(unpack_fmt, raw, cursor)[0]))
        cursor += uint_size

    chunks: list[bytes] = []
    for chunk_size in compressed_sizes:
        if chunk_size < 0 or cursor + chunk_size > len(raw):
            raise ValueError("compressed payload is truncated")
        chunks.append(zlib.decompress(raw[cursor : cursor + chunk_size]))
        cursor += chunk_size
    return b"".join(chunks)


def _read_vti_named_array(path: Path, *, name: str) -> np.ndarray:
    root = ET.parse(path).getroot()
    image_data = root.find(".//ImageData")
    if image_data is None:
        raise ValueError(f"ImageData is missing: {path}")

    appended = root.find(".//AppendedData")
    if appended is None or appended.text is None:
        raise ValueError(f"AppendedData is missing: {path}")
    encoded = "".join(appended.text.split())
    if encoded.startswith("_"):
        encoded = encoded[1:]

    arrays: list[tuple[int, ET.Element]] = []
    for data_array in root.findall(".//DataArray"):
        offset_raw = data_array.attrib.get("offset")
        if offset_raw is None:
            continue
        arrays.append((int(offset_raw), data_array))
    arrays.sort(key=lambda item: item[0])

    target_index = -1
    for idx, (_, data_array) in enumerate(arrays):
        if data_array.attrib.get("Name") == name:
            target_index = idx
            break
    if target_index < 0:
        raise ValueError(f"DataArray '{name}' not found in {path}")

    target_offset, target_array = arrays[target_index]
    next_offset = arrays[target_index + 1][0] if (target_index + 1) < len(arrays) else len(encoded)

    vtk_header_type = str(root.attrib.get("header_type", "UInt32"))
    vtk_compressor = str(root.attrib.get("compressor", ""))
    payload = _decode_appended_array(
        encoded_payload=encoded,
        offset=target_offset,
        next_offset=next_offset,
        header_type=vtk_header_type,
        compressor=vtk_compressor,
    )

    dtype = _dtype_from_vtk(str(target_array.attrib.get("type", "Float32")))
    flat = np.frombuffer(payload, dtype=dtype)

    point_shape = _extents_to_point_shape(image_data)
    cell_shape = _extents_to_cell_shape(image_data)
    point_count = int(point_shape[0] * point_shape[1] * point_shape[2])
    cell_count = int(cell_shape[0] * cell_shape[1] * cell_shape[2])

    if flat.size == point_count:
        nx, ny, nz = point_shape
        return flat.reshape((nz, ny, nx))
    if flat.size == cell_count:
        nx, ny, nz = cell_shape
        return flat.reshape((nz, ny, nx))
    raise ValueError(
        f"cannot infer grid shape for {name} in {path}: values={flat.size}, points={point_count}, cells={cell_count}"
    )


def _list_vti_paths(run_dir: Path, pattern: str) -> list[Path]:
    vti_paths = sorted(run_dir.glob(pattern))
    if not vti_paths:
        vti_paths = sorted(run_dir.glob("*.vti"))
    if not vti_paths:
        raise ValueError(f"no VTI files found under {run_dir}")
    return vti_paths


def _load_material_and_valid(
    *,
    vti_path: Path,
    valid_mask_array: str,
    legacy_mask_array: str,
    legacy_material_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        valid_raw = _read_vti_named_array(vti_path, name=valid_mask_array)
        valid = np.asarray(valid_raw > 0.5, dtype=bool)
    except Exception:
        valid = np.ones_like(_read_vti_named_array(vti_path, name="MaterialIds"), dtype=bool)

    try:
        material = np.asarray(_read_vti_named_array(vti_path, name="MaterialIds"), dtype=np.int32)
    except Exception:
        if str(legacy_mask_array).strip() == "":
            raise
        mask_raw = _read_vti_named_array(vti_path, name=str(legacy_mask_array))
        material = np.where(mask_raw > 0.5, int(legacy_material_id), 0).astype(np.int32)

    if material.shape != valid.shape:
        raise ValueError(f"material/valid shape mismatch in {vti_path}: {material.shape} vs {valid.shape}")
    return material, valid


def _select_target_material_id(
    *,
    data_dir: Path,
    run_ids: list[str],
    vti_pattern: str,
    valid_mask_array: str,
    legacy_mask_array: str,
    legacy_material_id: int,
    target_material_id: int | None,
    selection_mode: str,
    target_max_ratio: float,
    target_max_boundary_ratio: float,
    target_min_temporal_delta: float,
) -> tuple[int, dict[str, Any]]:
    aggregate: dict[int, dict[str, Any]] = {}
    run_records: list[dict[str, Any]] = []
    valid_counts: list[int] = []
    for run_id in run_ids:
        run_dir = data_dir / run_id
        vti_paths = _list_vti_paths(run_dir, vti_pattern)
        frame_records: list[dict[str, Any]] = []
        for frame_idx, path in enumerate(vti_paths):
            material, valid = _load_material_and_valid(
                vti_path=path,
                valid_mask_array=valid_mask_array,
                legacy_mask_array=legacy_mask_array,
                legacy_material_id=legacy_material_id,
            )
            valid_idx = np.argwhere(valid)
            if valid_idx.size > 0:
                z_min = int(np.min(valid_idx[:, 0]))
                z_max = int(np.max(valid_idx[:, 0]))
                y_min = int(np.min(valid_idx[:, 1]))
                y_max = int(np.max(valid_idx[:, 1]))
                x_min = int(np.min(valid_idx[:, 2]))
                x_max = int(np.max(valid_idx[:, 2]))
                boundary = np.zeros_like(valid, dtype=bool)
                boundary[z_min, :, :] = True
                boundary[z_max, :, :] = True
                boundary[:, y_min, :] = True
                boundary[:, y_max, :] = True
                boundary[:, :, x_min] = True
                boundary[:, :, x_max] = True
            else:
                boundary = np.zeros_like(valid, dtype=bool)
            material_valid = material[valid]
            valid_counts.append(int(material_valid.size))
            values, counts = np.unique(material_valid, return_counts=True)
            boundary_vals, boundary_counts = np.unique(material[valid & boundary], return_counts=True)
            boundary_map = {int(mat_id): int(cnt) for mat_id, cnt in zip(boundary_vals, boundary_counts)}
            by_material = {str(int(mat_id)): int(cnt) for mat_id, cnt in zip(values, counts)}
            frame_records.append(
                {
                    "frame_index": int(frame_idx),
                    "frame_path": str(path),
                    "counts_by_material": by_material,
                }
            )
            for mat_id, cnt in zip(values, counts):
                rec = aggregate.setdefault(
                    int(mat_id),
                    {
                        "counts": [],
                        "total_count": 0,
                        "boundary_count": 0,
                    },
                )
                rec["counts"].append(int(cnt))
                rec["total_count"] = int(rec["total_count"]) + int(cnt)
                rec["boundary_count"] = int(rec["boundary_count"]) + int(boundary_map.get(int(mat_id), 0))
        run_records.append(
            {
                "run_id": str(run_id),
                "num_frames": len(frame_records),
                "frames": frame_records,
            }
        )

    if not aggregate:
        raise ValueError("failed to collect MaterialIds counts from train runs")
    valid_mean = float(np.mean(np.asarray(valid_counts, dtype=np.float64))) if valid_counts else 1.0
    aggregate_summary: dict[str, dict[str, float]] = {}
    for mat_id, rec in aggregate.items():
        arr = np.asarray(rec.get("counts", []), dtype=np.float64)
        total_count = float(rec.get("total_count", 0))
        boundary_count = float(rec.get("boundary_count", 0))
        mean_count = float(np.mean(arr)) if arr.size > 0 else 0.0
        aggregate_summary[str(int(mat_id))] = {
            "count_min": float(np.min(arr)),
            "count_max": float(np.max(arr)),
            "count_mean": mean_count,
            "temporal_delta": float(np.max(arr) - np.min(arr)),
            "boundary_ratio": (boundary_count / total_count) if total_count > 0.0 else 0.0,
            "count_ratio_mean": (mean_count / valid_mean) if valid_mean > 0.0 else 0.0,
        }

    if target_material_id is not None:
        selected = int(target_material_id)
        if str(selected) not in aggregate_summary:
            raise ValueError(
                f"--target-material-id={selected} is not observed within ValidMask on train runs"
            )
        policy = "manual_material_id"
        reason = "explicit target-material-id"
    else:
        chosen_mode = str(selection_mode).strip().lower()
        selected = -1
        if chosen_mode == "auto_hole":
            candidates: list[tuple[int, float, float, float]] = []
            for mat_id_raw, stats in aggregate_summary.items():
                mat_id = int(mat_id_raw)
                delta = float(stats.get("temporal_delta", 0.0))
                ratio = float(stats.get("count_ratio_mean", 0.0))
                boundary_ratio = float(stats.get("boundary_ratio", 1.0))
                if delta < float(target_min_temporal_delta):
                    continue
                if ratio <= 0.0 or ratio > float(target_max_ratio):
                    continue
                if boundary_ratio > float(target_max_boundary_ratio):
                    continue
                score = delta * (1.0 - boundary_ratio)
                candidates.append((mat_id, score, delta, boundary_ratio))
            if candidates:
                candidates.sort(key=lambda row: (-row[1], -row[2], row[3], row[0]))
                selected = int(candidates[0][0])
                policy = "auto_hole_material"
                reason = (
                    "max temporal_delta*(1-boundary_ratio) with "
                    f"count_ratio<= {float(target_max_ratio):.4f} and "
                    f"boundary_ratio<= {float(target_max_boundary_ratio):.4f}"
                )
            else:
                chosen_mode = "auto_dynamic"
        if chosen_mode == "auto_dynamic":
            best_delta = -1.0
            for mat_id_raw, stats in aggregate_summary.items():
                delta = float(stats.get("temporal_delta", 0.0))
                mean_count = float(stats.get("count_mean", 0.0))
                if mean_count <= 0.0:
                    continue
                if delta > best_delta:
                    best_delta = delta
                    selected = int(mat_id_raw)
            if selected < 0:
                raise ValueError("auto material selection failed: no valid candidate found")
            policy = "auto_dynamic_material"
            reason = "max temporal_delta within ValidMask"

    diagnostics = {
        "selection_policy": policy,
        "selection_reason": reason,
        "selection_mode_requested": str(selection_mode),
        "selected_target_material_id": int(selected),
        "selection_constraints": {
            "target_max_ratio": float(target_max_ratio),
            "target_max_boundary_ratio": float(target_max_boundary_ratio),
            "target_min_temporal_delta": float(target_min_temporal_delta),
        },
        "aggregate_by_material": aggregate_summary,
        "train_runs": run_records,
    }
    return int(selected), diagnostics


def _build_phi_from_vti(
    *,
    vti_path: Path,
    target_material_id: int,
    valid_mask_array: str,
    legacy_mask_array: str,
    legacy_material_id: int,
    phi_boundary_clip_vox: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    material, valid = _load_material_and_valid(
        vti_path=vti_path,
        valid_mask_array=valid_mask_array,
        legacy_mask_array=legacy_mask_array,
        legacy_material_id=legacy_material_id,
    )
    valid_inner = _erode_valid_mask(valid, int(max(0, phi_boundary_clip_vox)))
    target_mask_valid = (material == int(target_material_id)) & valid
    target_mask = target_mask_valid & valid_inner
    outside_mask = (~target_mask) & valid_inner
    if _distance_transform_edt is not None:
        inside = _distance_transform_edt(target_mask)
        outside = _distance_transform_edt(outside_mask)
        phi = np.asarray(inside - outside, dtype=np.float32)
    else:
        phi = np.where(target_mask, 1.0, -1.0).astype(np.float32)
    if np.any(valid_inner):
        outside_fill = float(np.max(np.abs(phi[valid_inner])) + 1.0)
    else:
        outside_fill = 1.0
    phi = np.where(valid_inner, phi, outside_fill).astype(np.float32)
    return (
        phi,
        np.asarray(valid, dtype=bool),
        np.asarray(valid_inner, dtype=bool),
        int(np.count_nonzero(target_mask_valid)),
        int(np.count_nonzero(target_mask)),
    )


def _load_phi_frames_zyx(
    *,
    run_dir: Path,
    vti_pattern: str,
    target_material_id: int,
    valid_mask_array: str,
    mask_array: str,
    mask_material_id: int,
    phi_boundary_clip_vox: int,
) -> tuple[
    list[list[list[list[float]]]],
    list[np.ndarray],
    list[np.ndarray],
    list[str],
    tuple[int, int, int],
    int,
    int,
]:
    vti_paths = _list_vti_paths(run_dir, vti_pattern)

    frames: list[list[list[list[float]]]] = []
    valid_t: list[np.ndarray] = []
    valid_inner_t: list[np.ndarray] = []
    target_points_valid_total = 0
    target_points_inner_total = 0
    shape: tuple[int, int, int] | None = None
    for path in vti_paths:
        phi, valid, valid_inner, target_valid_count, target_inner_count = _build_phi_from_vti(
            vti_path=path,
            target_material_id=target_material_id,
            valid_mask_array=valid_mask_array,
            legacy_mask_array=mask_array,
            legacy_material_id=mask_material_id,
            phi_boundary_clip_vox=int(phi_boundary_clip_vox),
        )
        if shape is None:
            shape = (int(phi.shape[0]), int(phi.shape[1]), int(phi.shape[2]))
        elif phi.shape != shape:
            raise ValueError(f"inconsistent frame shape in run={run_dir.name}: {phi.shape} vs {shape}")
        frames.append(phi.tolist())
        valid_t.append(valid)
        valid_inner_t.append(valid_inner)
        target_points_valid_total += int(target_valid_count)
        target_points_inner_total += int(target_inner_count)

    assert shape is not None
    return (
        frames,
        valid_t,
        valid_inner_t,
        [str(path) for path in vti_paths],
        shape,
        int(target_points_valid_total),
        int(target_points_inner_total),
    )


@dataclass(frozen=True)
class PreparedRun:
    run_id: str
    dt: float
    recipe: dict[str, float]
    phi_t: list[list[list[list[float]]]]
    valid_t: list[np.ndarray]
    valid_inner_t: list[np.ndarray]
    target_points_valid_total: int
    target_points_inner_total: int
    frame_paths: list[str]
    frame_shape_zyx: tuple[int, int, int]


def _prepare_one_run(
    *,
    data_dir: Path,
    run_dir_name: str,
    recipe: dict[str, float],
    dt: float,
    vti_pattern: str,
    target_material_id: int,
    valid_mask_array: str,
    mask_array: str,
    mask_material_id: int,
    phi_boundary_clip_vox: int,
) -> PreparedRun:
    run_dir = data_dir / run_dir_name
    if not run_dir.exists() or not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")

    (
        frames_zyx,
        valid_t,
        valid_inner_t,
        frame_paths,
        frame_shape_zyx,
        target_points_valid_total,
        target_points_inner_total,
    ) = _load_phi_frames_zyx(
        run_dir=run_dir,
        vti_pattern=vti_pattern,
        target_material_id=target_material_id,
        valid_mask_array=valid_mask_array,
        mask_array=mask_array,
        mask_material_id=mask_material_id,
        phi_boundary_clip_vox=int(phi_boundary_clip_vox),
    )
    if len(frames_zyx) < 2:
        raise ValueError(f"need at least 2 frames for training target: {run_dir}")
    return PreparedRun(
        run_id=str(run_dir_name),
        dt=float(dt),
        recipe={str(k): float(v) for k, v in recipe.items()},
        phi_t=frames_zyx,
        valid_t=valid_t,
        valid_inner_t=valid_inner_t,
        target_points_valid_total=int(target_points_valid_total),
        target_points_inner_total=int(target_points_inner_total),
        frame_paths=frame_paths,
        frame_shape_zyx=frame_shape_zyx,
    )


def _laplacian_proxy_3d(phi: np.ndarray) -> np.ndarray:
    pad = np.pad(phi, ((1, 1), (1, 1), (1, 1)), mode="edge")
    center = pad[1:-1, 1:-1, 1:-1]
    left = pad[1:-1, 1:-1, :-2]
    right = pad[1:-1, 1:-1, 2:]
    up = pad[1:-1, :-2, 1:-1]
    down = pad[1:-1, 2:, 1:-1]
    prev_z = pad[:-2, 1:-1, 1:-1]
    next_z = pad[2:, 1:-1, 1:-1]
    return (left + right + up + down + prev_z + next_z) - (6.0 * center)


def _erode_valid_mask(valid: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return np.asarray(valid, dtype=bool)
    base = np.asarray(valid, dtype=bool)
    if _distance_transform_edt is not None:
        return base & (_distance_transform_edt(base) > float(margin))
    current = base
    for _ in range(int(margin)):
        pad = np.pad(current, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=False)
        current = (
            pad[1:-1, 1:-1, 1:-1]
            & pad[:-2, 1:-1, 1:-1]
            & pad[2:, 1:-1, 1:-1]
            & pad[1:-1, :-2, 1:-1]
            & pad[1:-1, 2:, 1:-1]
            & pad[1:-1, 1:-1, :-2]
            & pad[1:-1, 1:-1, 2:]
        )
    return current


def _edge_ratio_xy(coords_xyz: np.ndarray, valid: np.ndarray, margin: int) -> float:
    if coords_xyz.size == 0:
        return 0.0
    coords_zyx = coords_xyz[:, [2, 1, 0]].astype(np.int32, copy=False)
    valid_idx = np.argwhere(valid)
    if valid_idx.size == 0:
        return 0.0
    y_min = int(np.min(valid_idx[:, 1]))
    y_max = int(np.max(valid_idx[:, 1]))
    x_min = int(np.min(valid_idx[:, 2]))
    x_max = int(np.max(valid_idx[:, 2]))
    m = int(max(1, margin))
    near = (
        (coords_zyx[:, 1] <= (y_min + m))
        | (coords_zyx[:, 1] >= (y_max - m))
        | (coords_zyx[:, 2] <= (x_min + m))
        | (coords_zyx[:, 2] >= (x_max - m))
    )
    return float(np.mean(near))


def _edge_ratio_z(coords_xyz: np.ndarray, valid: np.ndarray, margin: int) -> float:
    if coords_xyz.size == 0:
        return 0.0
    coords_zyx = coords_xyz[:, [2, 1, 0]].astype(np.int32, copy=False)
    valid_idx = np.argwhere(valid)
    if valid_idx.size == 0:
        return 0.0
    z_min = int(np.min(valid_idx[:, 0]))
    z_max = int(np.max(valid_idx[:, 0]))
    near = (coords_zyx[:, 0] <= (z_min + int(max(1, margin)))) | (coords_zyx[:, 0] >= (z_max - int(max(1, margin))))
    return float(np.mean(near))


def _build_narrow_band_run(
    *,
    run: PreparedRun,
    band_width: float,
    min_grad_norm: float,
    include_terminal_step_target: bool,
    domain_boundary_margin_vox: int,
) -> tuple[NarrowBandRun, dict[str, Any]]:
    steps: list[NarrowBandStep] = []
    step_diagnostics: list[dict[str, Any]] = []
    total_points = 0
    weighted_edge_sum = 0.0
    step_count = len(run.phi_t)
    last_step = step_count - 1
    for step_idx in range(step_count):
        if (not include_terminal_step_target) and step_idx >= last_step:
            continue
        next_idx = min(step_idx + 1, step_count - 1)
        phi = np.asarray(run.phi_t[step_idx], dtype=np.float32)
        phi_next = np.asarray(run.phi_t[next_idx], dtype=np.float32)
        valid = np.asarray(run.valid_t[step_idx], dtype=bool)
        valid_inner = np.asarray(run.valid_inner_t[step_idx], dtype=bool)
        valid_inner = _erode_valid_mask(valid_inner, int(domain_boundary_margin_vox))
        valid_idx = np.argwhere(valid_inner)
        if valid_idx.size > 0:
            z_min = int(np.min(valid_idx[:, 0]))
            z_max = int(np.max(valid_idx[:, 0]))
            y_min = int(np.min(valid_idx[:, 1]))
            y_max = int(np.max(valid_idx[:, 1]))
            x_min = int(np.min(valid_idx[:, 2]))
            x_max = int(np.max(valid_idx[:, 2]))
        else:
            z_min = z_max = y_min = y_max = x_min = x_max = 0
        boundary_margin = int(max(0, domain_boundary_margin_vox))

        edge_order = 2 if all(int(dim) >= 3 for dim in phi.shape) else 1
        grads = np.gradient(phi, edge_order=edge_order)
        grad_norm = np.sqrt((grads[0] ** 2) + (grads[1] ** 2) + (grads[2] ** 2))
        curvature = _laplacian_proxy_3d(phi)

        mask = (
            (np.abs(phi) <= float(band_width))
            & (grad_norm >= float(min_grad_norm))
            & valid_inner
        )
        coords_zyx = np.argwhere(mask)
        if coords_zyx.size > 0 and boundary_margin > 0:
            keep = (
                (coords_zyx[:, 0] > (z_min + boundary_margin))
                & (coords_zyx[:, 0] < (z_max - boundary_margin))
                & (coords_zyx[:, 1] > (y_min + boundary_margin))
                & (coords_zyx[:, 1] < (y_max - boundary_margin))
                & (coords_zyx[:, 2] > (x_min + boundary_margin))
                & (coords_zyx[:, 2] < (x_max - boundary_margin))
            )
            coords_zyx = coords_zyx[keep]
        if coords_zyx.size == 0:
            steps.append(NarrowBandStep(coords=[], feat=[], vn_target=[]))
            step_diagnostics.append(
                {
                    "step_index": int(step_idx),
                    "num_points": 0,
                    "edge_point_ratio_xy": 0.0,
                    "edge_point_ratio_z": 0.0,
                    "z_min": None,
                    "z_max": None,
                }
            )
            continue

        coords_xyz = coords_zyx[:, [2, 1, 0]].astype(np.int32, copy=False)
        gnorm = grad_norm[coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]]
        phi_vals = phi[coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]]
        phi_next_vals = phi_next[coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]]
        curv_vals = curvature[coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]]
        vn_target = ((phi_vals - phi_next_vals) / (float(run.dt) * gnorm)).reshape(-1, 1)
        feat = np.stack(
            [phi_vals, gnorm, curv_vals, np.abs(phi_vals)],
            axis=1,
        )
        steps.append(
            NarrowBandStep(
                coords=coords_xyz.tolist(),
                feat=feat.astype(np.float32, copy=False).tolist(),
                vn_target=vn_target.astype(np.float32, copy=False).tolist(),
            )
        )
        point_count = int(coords_xyz.shape[0])
        ratio_xy = _edge_ratio_xy(coords_xyz, valid_inner, int(domain_boundary_margin_vox))
        ratio_z = _edge_ratio_z(coords_xyz, valid_inner, int(domain_boundary_margin_vox))
        weighted_edge_sum += (ratio_xy * float(point_count))
        total_points += point_count
        step_diagnostics.append(
            {
                "step_index": int(step_idx),
                "num_points": point_count,
                "edge_point_ratio_xy": float(ratio_xy),
                "edge_point_ratio_z": float(ratio_z),
                "z_min": int(np.min(coords_xyz[:, 2])),
                "z_max": int(np.max(coords_xyz[:, 2])),
            }
        )

    return (
        NarrowBandRun(
            run_id=str(run.run_id),
            recipe=[float(v) for _, v in sorted(run.recipe.items())],
            dt=float(run.dt),
            steps=steps,
        ),
        {
            "run_id": str(run.run_id),
            "num_points_total": int(total_points),
            "edge_point_ratio_xy": (float(weighted_edge_sum) / float(total_points)) if total_points > 0 else 0.0,
            "steps": step_diagnostics,
        },
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare-dataset-3d-test2",
        description="Prepare ataset_3d_test2 into train narrow-band HDF5 + holdout JSON.",
    )
    parser.add_argument("--data-dir", default="ataset_3d_test2", help="Input dataset root directory.")
    parser.add_argument(
        "--train-runs",
        default=",".join([f"run_{idx:04d}" for idx in range(12)]),
        help="Comma-separated train run directory names.",
    )
    parser.add_argument(
        "--holdout-runs",
        default="run_0012",
        help="Comma-separated holdout run directory names.",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/dataset_3d_test2_pilot/prepared",
        help="Output directory for prepared artifacts.",
    )
    parser.add_argument("--vti-pattern", default="vox_t*.vti", help="VTI glob pattern inside each run directory.")
    parser.add_argument("--band-width", type=float, default=0.5, help="Narrow-band width.")
    parser.add_argument("--min-grad-norm", type=float, default=1e-6, help="Minimum grad norm for targets.")
    parser.add_argument(
        "--target-material-id",
        type=int,
        default=None,
        help="Target MaterialIds value. If omitted, selects the most dynamic material within ValidMask.",
    )
    parser.add_argument(
        "--target-selection-mode",
        choices=("auto_hole", "auto_dynamic"),
        default="auto_hole",
        help="Automatic target material policy when --target-material-id is not provided.",
    )
    parser.add_argument(
        "--target-max-ratio",
        type=float,
        default=0.2,
        help="Upper bound for count_ratio_mean in auto_hole selection.",
    )
    parser.add_argument(
        "--target-max-boundary-ratio",
        type=float,
        default=0.2,
        help="Upper bound for boundary_ratio in auto_hole selection.",
    )
    parser.add_argument(
        "--target-min-temporal-delta",
        type=float,
        default=1.0,
        help="Lower bound for temporal_delta in auto_hole selection.",
    )
    parser.add_argument(
        "--valid-mask-array",
        default="ValidMask",
        help="VTI mask array that defines valid simulation domain.",
    )
    parser.add_argument(
        "--domain-boundary-margin-vox",
        type=int,
        default=1,
        help="Exclude samples within this voxel margin from domain boundary.",
    )
    parser.add_argument(
        "--phi-boundary-clip-vox",
        type=int,
        default=3,
        help="Erode ValidMask by this margin before computing SDF to suppress fixed domain boundaries.",
    )
    parser.add_argument(
        "--include-terminal-step-target",
        action="store_true",
        help="Include terminal step target (defaults to excluded).",
    )
    parser.add_argument(
        "--mask-array",
        default="",
        help="Legacy mask array fallback when MaterialIds is unavailable.",
    )
    parser.add_argument(
        "--mask-material-id",
        type=int,
        default=2,
        help="Legacy material id fallback when mask-array path is used.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"data-dir does not exist: {data_dir}")
    index_path = data_dir / "index.csv"
    if not index_path.exists():
        raise ValueError(f"index.csv missing: {index_path}")

    index_rows = _load_index_rows(index_path)
    row_by_dir = {str(row.get("run_dir", "")).strip(): row for row in index_rows}
    train_ids = _parse_run_list(args.train_runs)
    holdout_ids = _parse_run_list(args.holdout_runs)
    if not train_ids:
        raise ValueError("train-runs must not be empty")
    if not holdout_ids:
        raise ValueError("holdout-runs must not be empty")
    overlap = sorted(set(train_ids).intersection(set(holdout_ids)))
    if overlap:
        raise ValueError(f"train/holdout overlap is not allowed: {overlap}")

    recipes_by_run: dict[str, dict[str, float]] = {}
    dt_by_run: dict[str, float] = {}
    missing_from_index: list[str] = []

    for run_dir_name in [*train_ids, *holdout_ids]:
        row = row_by_dir.get(run_dir_name)
        if row is None:
            missing_from_index.append(run_dir_name)
            continue
        run_dir = data_dir / run_dir_name
        inputs_path = run_dir / "inputs.json"
        if not inputs_path.exists():
            raise ValueError(f"inputs.json missing: {inputs_path}")

        recipe_index = _numeric_recipe_from_index(row)
        recipe_inputs, dt = _numeric_recipe_from_inputs(inputs_path)
        merged = dict(recipe_index)
        for key, value in recipe_inputs.items():
            merged.setdefault(str(key), float(value))
        recipes_by_run[run_dir_name] = {str(k): float(v) for k, v in merged.items()}
        dt_by_run[run_dir_name] = float(dt)

    if missing_from_index:
        raise ValueError(f"run_dir missing in index.csv: {missing_from_index}")

    selected_runs = [*train_ids, *holdout_ids]
    key_sets = [set(recipes_by_run[run_id].keys()) for run_id in selected_runs]
    shared_keys = sorted(set.intersection(*key_sets)) if key_sets else []
    if not shared_keys:
        raise ValueError("no shared numeric recipe keys found across selected runs")

    selected_target_material_id, material_diagnostics = _select_target_material_id(
        data_dir=data_dir,
        run_ids=train_ids,
        vti_pattern=str(args.vti_pattern),
        valid_mask_array=str(args.valid_mask_array),
        legacy_mask_array=str(args.mask_array),
        legacy_material_id=int(args.mask_material_id),
        target_material_id=args.target_material_id,
        selection_mode=str(args.target_selection_mode),
        target_max_ratio=float(args.target_max_ratio),
        target_max_boundary_ratio=float(args.target_max_boundary_ratio),
        target_min_temporal_delta=float(args.target_min_temporal_delta),
    )
    print(
        f"[prepare] selected target material id={selected_target_material_id} "
        f"policy={material_diagnostics.get('selection_policy')}",
        flush=True,
    )

    train_nb_runs: list[NarrowBandRun] = []
    train_sampling_records: list[dict[str, Any]] = []
    train_audit_runs: list[dict[str, Any]] = []
    holdout_runs: list[SyntheticSDFRun] = []
    holdout_audit_runs: list[dict[str, Any]] = []

    for run_id in train_ids:
        print(f"[prepare] train run {run_id} ...", flush=True)
        normalized_recipe = {key: float(recipes_by_run[run_id].get(key, 0.0)) for key in shared_keys}
        prepared = _prepare_one_run(
            data_dir=data_dir,
            run_dir_name=run_id,
            recipe=normalized_recipe,
            dt=float(dt_by_run[run_id]),
            vti_pattern=str(args.vti_pattern),
            target_material_id=int(selected_target_material_id),
            valid_mask_array=str(args.valid_mask_array),
            mask_array=str(args.mask_array),
            mask_material_id=int(args.mask_material_id),
            phi_boundary_clip_vox=max(0, int(args.phi_boundary_clip_vox)),
        )
        nb_run, sampling_diag = _build_narrow_band_run(
            run=prepared,
            band_width=float(args.band_width),
            min_grad_norm=float(args.min_grad_norm),
            include_terminal_step_target=bool(args.include_terminal_step_target),
            domain_boundary_margin_vox=max(0, int(args.domain_boundary_margin_vox)),
        )
        train_nb_runs.append(nb_run)
        train_sampling_records.append(sampling_diag)
        train_audit_runs.append(
            {
                "run_id": prepared.run_id,
                "dt": prepared.dt,
                "recipe": prepared.recipe,
                "frame_paths": prepared.frame_paths,
                "num_frames": len(prepared.phi_t),
                "frame_shape_zyx": list(prepared.frame_shape_zyx),
                "target_points_valid_total": int(prepared.target_points_valid_total),
                "target_points_inner_total": int(prepared.target_points_inner_total),
            }
        )

    for run_id in holdout_ids:
        print(f"[prepare] holdout run {run_id} ...", flush=True)
        normalized_recipe = {key: float(recipes_by_run[run_id].get(key, 0.0)) for key in shared_keys}
        prepared = _prepare_one_run(
            data_dir=data_dir,
            run_dir_name=run_id,
            recipe=normalized_recipe,
            dt=float(dt_by_run[run_id]),
            vti_pattern=str(args.vti_pattern),
            target_material_id=int(selected_target_material_id),
            valid_mask_array=str(args.valid_mask_array),
            mask_array=str(args.mask_array),
            mask_material_id=int(args.mask_material_id),
            phi_boundary_clip_vox=max(0, int(args.phi_boundary_clip_vox)),
        )
        holdout_runs.append(
            SyntheticSDFRun(
                run_id=prepared.run_id,
                dt=prepared.dt,
                recipe=prepared.recipe,
                phi_t=prepared.phi_t,
            )
        )
        holdout_audit_runs.append(
            {
                "run_id": prepared.run_id,
                "dt": prepared.dt,
                "recipe": prepared.recipe,
                "frame_paths": prepared.frame_paths,
                "num_frames": len(prepared.phi_t),
                "frame_shape_zyx": list(prepared.frame_shape_zyx),
                "target_points_valid_total": int(prepared.target_points_valid_total),
                "target_points_inner_total": int(prepared.target_points_inner_total),
            }
        )

    if len(train_nb_runs) != len(train_ids):
        raise ValueError("prepared train runs do not match requested train-runs")
    if len(holdout_runs) != len(holdout_ids):
        raise ValueError("prepared holdout runs do not match requested holdout-runs")

    nb_dataset = NarrowBandDataset(runs=train_nb_runs)
    holdout_dataset = SyntheticSDFDataset(runs=holdout_runs)

    train_dataset_path = out_dir / "train_dataset.json"
    holdout_dataset_path = out_dir / "holdout_dataset.json"
    split_manifest_path = out_dir / "split_manifest.json"
    train_h5_path = out_dir / "train_narrow_band.h5"
    feature_contract_path = out_dir / "feature_contract.json"
    point_manifest_path = out_dir / "point_level_manifest.json"
    material_diag_path = out_dir / "material_diagnostics.json"
    sampling_diag_path = out_dir / "sampling_diagnostics.json"

    feature_names = [
        "phi",
        "grad_norm",
        "curvature_proxy",
        "band_distance",
        "coord_x",
        "coord_y",
        "coord_z",
        "step_index",
    ]
    feature_contract = {
        "recipe_keys": list(shared_keys),
        "feature_names": feature_names,
        "cond_dim": len(shared_keys),
        "feat_dim": len(feature_names),
        "band_width": float(args.band_width),
        "min_grad_norm": float(args.min_grad_norm),
    }
    point_count_total = int(sum(len(step.coords) for run in train_nb_runs for step in run.steps))
    step_count_total = int(sum(len(run.steps) for run in train_nb_runs))
    point_manifest = {
        "schema_version": "1",
        "target_mode": "vn_narrow_band",
        "num_rows": point_count_total,
        "num_targets": point_count_total,
        "num_steps_total": step_count_total,
        "sample_ref_fields": ["run_id", "step_index", "point_index", "sample_index"],
        "feature_contract": feature_contract,
        "terminal_target_policy": "include" if bool(args.include_terminal_step_target) else "exclude",
        "selected_target_material_id": int(selected_target_material_id),
    }

    points_per_step: dict[str, int] = {}
    edge_weighted_sum_xy = 0.0
    edge_weighted_sum_z = 0.0
    edge_weighted_count = 0
    points_per_run: dict[str, int] = {}
    target_points_valid_total = 0
    target_points_inner_total = 0
    for run_record in train_sampling_records:
        run_points = int(run_record.get("num_points_total", 0))
        run_ratio_xy = float(run_record.get("edge_point_ratio_xy", 0.0))
        edge_weighted_sum_xy += (run_ratio_xy * float(run_points))
        edge_weighted_count += run_points
        run_id = str(run_record.get("run_id", ""))
        points_per_run[run_id] = run_points
        maybe_run = next((r for r in train_audit_runs if str(r.get("run_id", "")) == run_id), None)
        if maybe_run:
            target_points_valid_total += int(maybe_run.get("target_points_valid_total", 0))
            target_points_inner_total += int(maybe_run.get("target_points_inner_total", 0))
        for step_record in run_record.get("steps", []):
            step_idx = str(step_record.get("step_index", ""))
            step_points = int(step_record.get("num_points", 0))
            points_per_step[step_idx] = int(points_per_step.get(step_idx, 0) + step_points)
            edge_weighted_sum_z += float(step_record.get("edge_point_ratio_z", 0.0)) * float(step_points)
    sampling_diagnostics = {
        "num_points_total": int(sum(points_per_run.values())),
        "edge_point_ratio_xy": (float(edge_weighted_sum_xy) / float(edge_weighted_count)) if edge_weighted_count > 0 else 0.0,
        "edge_point_ratio_z": (float(edge_weighted_sum_z) / float(edge_weighted_count)) if edge_weighted_count > 0 else 0.0,
        "target_points_in_inner_ratio": (float(target_points_inner_total) / float(target_points_valid_total)) if target_points_valid_total > 0 else 0.0,
        "points_per_run": points_per_run,
        "points_per_step": points_per_step,
        "domain_boundary_margin_vox": int(max(0, args.domain_boundary_margin_vox)),
        "phi_boundary_clip_vox": int(max(0, args.phi_boundary_clip_vox)),
        "domain_clip_policy": "exclude_outside_valid",
        "selected_target_material_id": int(selected_target_material_id),
        "runs": train_sampling_records,
    }

    _write_json(
        train_dataset_path,
        {
            "format": "dataset_3d_test2_train_manifest_v1",
            "note": "train phi_t is intentionally omitted to avoid oversized JSON; use train_narrow_band.h5",
            "runs": train_audit_runs,
        },
    )
    _write_json(holdout_dataset_path, holdout_dataset.to_dict())
    write_hdf5_dataset(train_h5_path, nb_dataset)
    _write_json(feature_contract_path, feature_contract)
    _write_json(point_manifest_path, point_manifest)
    _write_json(material_diag_path, material_diagnostics)
    _write_json(sampling_diag_path, sampling_diagnostics)

    split_manifest = {
        "data_dir": str(data_dir),
        "train_runs": [rid for rid in train_ids],
        "holdout_runs": [run.run_id for run in holdout_dataset.runs],
        "num_train_runs": len(train_ids),
        "num_holdout_runs": len(holdout_runs),
        "recipe_keys": shared_keys,
        "band_width": float(args.band_width),
        "min_grad_norm": float(args.min_grad_norm),
        "terminal_target_policy": "include" if bool(args.include_terminal_step_target) else "exclude",
        "selected_target_material_id": int(selected_target_material_id),
        "phi_source_policy": str(material_diagnostics.get("selection_policy", "auto_dynamic_material")),
        "domain_clip_policy": "exclude_outside_valid",
        "domain_boundary_margin_vox": int(max(0, args.domain_boundary_margin_vox)),
        "phi_boundary_clip_vox": int(max(0, args.phi_boundary_clip_vox)),
        "phi_domain_policy": "inner_valid_sdf",
        "vti_pattern": str(args.vti_pattern),
        "phi_source": {
            "valid_mask_array": str(args.valid_mask_array),
            "mask_array": str(args.mask_array),
            "mask_material_id_fallback": int(args.mask_material_id),
            "sdf_backend": "scipy_distance_transform_edt" if _distance_transform_edt is not None else "binary_sign_fallback",
        },
        "train_runs_audit": train_audit_runs,
        "holdout_runs_audit": holdout_audit_runs,
        "artifacts": {
            "train_dataset_json": str(train_dataset_path),
            "holdout_dataset_json": str(holdout_dataset_path),
            "train_narrow_band_h5": str(train_h5_path),
            "feature_contract_json": str(feature_contract_path),
            "point_level_manifest_json": str(point_manifest_path),
            "material_diagnostics_json": str(material_diag_path),
            "sampling_diagnostics_json": str(sampling_diag_path),
        },
    }
    _write_json(split_manifest_path, split_manifest)

    print(f"prepared train dataset: {train_dataset_path}")
    print(f"prepared holdout dataset: {holdout_dataset_path}")
    print(f"prepared narrow-band h5: {train_h5_path}")
    print(f"feature contract: {feature_contract_path}")
    print(f"point-level manifest: {point_manifest_path}")
    print(f"material diagnostics: {material_diag_path}")
    print(f"sampling diagnostics: {sampling_diag_path}")
    print(f"split manifest: {split_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
