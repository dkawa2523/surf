#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from wafer_surrogate.core.rollout import rollout
from wafer_surrogate.data.synthetic import SyntheticSDFDataset, SyntheticSDFRun
from wafer_surrogate.metrics import compute_temporal_diagnostics
from wafer_surrogate.models.api import make_model
from wafer_surrogate.models.sparse_unet_film import (
    OptionalSparseDependencyUnavailable,
    SparseTensorVnModel,
)
from wafer_surrogate.runtime.capabilities import detect_runtime_capabilities
from wafer_surrogate.viz.utils import (
    load_pyplot,
    resolve_visualization_config,
    viz_enabled,
    write_visualization_manifest,
)

try:
    import prepare_dataset_3d_test2 as _prepare_utils
except Exception:  # pragma: no cover - optional local helper
    _prepare_utils = None

try:
    from scipy.ndimage import distance_transform_edt as _distance_transform_edt
except Exception:  # pragma: no cover - optional dependency
    _distance_transform_edt = None


def _to_float_nested(frame: Any) -> Any:
    if hasattr(frame, "tolist"):
        frame = frame.tolist()
    if isinstance(frame, Sequence) and not isinstance(frame, (str, bytes, bytearray)):
        return [_to_float_nested(cell) for cell in frame]
    return float(frame)


def _flatten_values(frame: Any) -> list[float]:
    if hasattr(frame, "tolist"):
        frame = frame.tolist()
    if isinstance(frame, Sequence) and not isinstance(frame, (str, bytes, bytearray)):
        out: list[float] = []
        for item in frame:
            out.extend(_flatten_values(item))
        return out
    return [float(frame)]


def _frame_mean(frame: Any) -> float:
    values = _flatten_values(frame)
    if not values:
        raise ValueError("frame is empty")
    return float(sum(values) / float(len(values)))


def _to_numpy_zyx(frame: Any) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"expected 2D/3D frame, got shape={arr.shape}")
    return arr


def _load_dataset(path: Path) -> SyntheticSDFDataset:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, Mapping):
        raise ValueError(f"holdout-json must be a mapping: {path}")
    runs_payload = payload.get("runs")
    if not isinstance(runs_payload, list) or not runs_payload:
        raise ValueError(f"holdout-json must contain non-empty runs: {path}")
    runs: list[SyntheticSDFRun] = []
    for idx, run in enumerate(runs_payload):
        if not isinstance(run, Mapping):
            raise ValueError(f"runs[{idx}] must be a mapping")
        run_id = str(run.get("run_id", f"holdout_{idx:03d}"))
        dt = float(run.get("dt", 1.0))
        recipe_raw = run.get("recipe", {})
        if not isinstance(recipe_raw, Mapping):
            raise ValueError(f"runs[{idx}].recipe must be mapping")
        recipe = {str(k): float(v) for k, v in recipe_raw.items()}
        phi_t_raw = run.get("phi_t")
        if not isinstance(phi_t_raw, list) or not phi_t_raw:
            raise ValueError(f"runs[{idx}].phi_t must be non-empty list")
        phi_t = [_to_float_nested(frame) for frame in phi_t_raw]
        runs.append(SyntheticSDFRun(run_id=run_id, dt=dt, recipe=recipe, phi_t=phi_t))
    return SyntheticSDFDataset(runs=runs)


def _load_split_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.parent / "split_manifest.json"
    if not manifest_path.exists() or not manifest_path.is_file():
        return {}
    with manifest_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, Mapping):
        return {}
    return {str(k): v for k, v in payload.items()}


def _frame_paths_by_run(split_manifest: Mapping[str, Any]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for key in ("holdout_runs_audit", "train_runs_audit"):
        runs = split_manifest.get(key)
        if not isinstance(runs, list):
            continue
        for row in runs:
            if not isinstance(row, Mapping):
                continue
            run_id = str(row.get("run_id", "")).strip()
            frames = row.get("frame_paths")
            if run_id == "" or not isinstance(frames, list):
                continue
            out[run_id] = [Path(str(p)) for p in frames if str(p).strip() != ""]
    return out


def _load_valid_masks_from_paths(
    *,
    frame_paths: Sequence[Path],
    valid_mask_array: str,
    fallback_shape: tuple[int, int, int],
) -> list[np.ndarray]:
    default_masks = [np.ones(fallback_shape, dtype=bool) for _ in frame_paths]
    if _prepare_utils is None or not hasattr(_prepare_utils, "_read_vti_named_array"):
        return default_masks
    reader = getattr(_prepare_utils, "_read_vti_named_array")
    out: list[np.ndarray] = []
    for idx, path in enumerate(frame_paths):
        try:
            arr = reader(path, name=str(valid_mask_array))
            valid = np.asarray(arr > 0.5, dtype=bool)
            if valid.shape != fallback_shape:
                valid = np.ones(fallback_shape, dtype=bool)
        except Exception:
            valid = np.ones(fallback_shape, dtype=bool)
        out.append(valid)
    if len(out) < len(frame_paths):
        out.extend(default_masks[len(out) :])
    return out


def _load_material_masks_from_paths(
    *,
    frame_paths: Sequence[Path],
    target_material_id: int,
    fallback_shape: tuple[int, int, int],
) -> list[np.ndarray]:
    default_masks = [np.zeros(fallback_shape, dtype=bool) for _ in frame_paths]
    if _prepare_utils is None or not hasattr(_prepare_utils, "_read_vti_named_array"):
        return default_masks
    reader = getattr(_prepare_utils, "_read_vti_named_array")
    out: list[np.ndarray] = []
    for path in frame_paths:
        try:
            mat = np.asarray(reader(path, name="MaterialIds"), dtype=np.int32)
            mask = np.asarray(mat == int(target_material_id), dtype=bool)
            if mask.shape != fallback_shape:
                mask = np.zeros(fallback_shape, dtype=bool)
        except Exception:
            mask = np.zeros(fallback_shape, dtype=bool)
        out.append(mask)
    if len(out) < len(frame_paths):
        out.extend(default_masks[len(out) :])
    return out


def _erode_mask(mask: np.ndarray, margin: int) -> np.ndarray:
    base = np.asarray(mask, dtype=bool)
    if int(margin) <= 0:
        return base
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


def _shrink_analysis_mask(
    valid_mask: np.ndarray,
    *,
    xy_margin_vox: int,
    z_margin_vox: int,
) -> np.ndarray:
    mask = np.asarray(valid_mask, dtype=bool)
    idx = np.argwhere(mask)
    if idx.size == 0:
        return mask
    z_min = int(np.min(idx[:, 0]))
    z_max = int(np.max(idx[:, 0]))
    y_min = int(np.min(idx[:, 1]))
    y_max = int(np.max(idx[:, 1]))
    x_min = int(np.min(idx[:, 2]))
    x_max = int(np.max(idx[:, 2]))
    m_xy = int(max(0, xy_margin_vox))
    m_z = int(max(0, z_margin_vox))
    z0 = z_min + m_z
    z1 = z_max - m_z
    y0 = y_min + m_xy
    y1 = y_max - m_xy
    x0 = x_min + m_xy
    x1 = x_max - m_xy
    if z0 > z1 or y0 > y1 or x0 > x1:
        return np.zeros_like(mask, dtype=bool)
    roi = np.zeros_like(mask, dtype=bool)
    roi[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = True
    return mask & roi


def _resolve_checkpoint(path: Path, checkpoint_raw: str) -> Path:
    candidate = Path(checkpoint_raw)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (path.parent / candidate).resolve()


def _load_model_from_state(path: Path) -> tuple[Any, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, Mapping):
        raise ValueError(f"model-state must be mapping: {path}")
    backend = str(payload.get("model_backend", "")).strip().lower()
    if backend == "sparse_tensor_checkpoint":
        checkpoint_raw = payload.get("checkpoint_path")
        if checkpoint_raw is None:
            raise ValueError("model-state backend sparse_tensor_checkpoint requires checkpoint_path")
        checkpoint = _resolve_checkpoint(path, str(checkpoint_raw))
        if not checkpoint.exists() or not checkpoint.is_file():
            raise ValueError(f"sparse checkpoint does not exist: {checkpoint}")
        try:
            model = SparseTensorVnModel.from_checkpoint(
                checkpoint,
                device=str(payload.get("device", "cpu")),
            )
        except OptionalSparseDependencyUnavailable as exc:
            raise ValueError(f"sparse checkpoint load failed due to missing dependency: {exc}") from exc
        return model, {str(k): v for k, v in payload.items()}

    model_name = str(payload.get("model_name", "baseline_vn_linear_trainable"))
    model = make_model(model_name)
    state = payload.get("state")
    if isinstance(state, Mapping):
        for key, value in state.items():
            setattr(model, str(key), value)
    return model, {str(k): v for k, v in payload.items()}


def _final_frame_l1(pred_frame: Any, ref_frame: Any) -> float:
    pred_vals = _flatten_values(pred_frame)
    ref_vals = _flatten_values(ref_frame)
    n = min(len(pred_vals), len(ref_vals))
    if n == 0:
        return 0.0
    return float(sum(abs(float(pred_vals[i]) - float(ref_vals[i])) for i in range(n)) / float(n))


def _vn_frame_mean_mae(pred_phi_t: list[Any], ref_phi_t: list[Any], dt: float) -> float:
    step_count = min(len(pred_phi_t), len(ref_phi_t))
    if step_count < 2:
        return 0.0
    errors: list[float] = []
    for step_idx in range(step_count - 1):
        pred_vn = (_frame_mean(pred_phi_t[step_idx]) - _frame_mean(pred_phi_t[step_idx + 1])) / float(dt)
        ref_vn = (_frame_mean(ref_phi_t[step_idx]) - _frame_mean(ref_phi_t[step_idx + 1])) / float(dt)
        errors.append(abs(float(pred_vn) - float(ref_vn)))
    return float(fmean(errors)) if errors else 0.0


def _write_temporal_plot(*, run_id: str, pred_phi_t: list[Any], ref_phi_t: list[Any], out_dir: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    step_count = min(len(pred_phi_t), len(ref_phi_t))
    if step_count < 1:
        return None
    steps = list(range(step_count))
    pred_means = [_frame_mean(pred_phi_t[idx]) for idx in steps]
    ref_means = [_frame_mean(ref_phi_t[idx]) for idx in steps]
    fig = plt.figure(figsize=(6.0, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(steps, ref_means, label="reference", linewidth=2.0)
    ax.plot(steps, pred_means, label="prediction", linewidth=2.0)
    ax.set_xlabel("step")
    ax.set_ylabel("phi_mean")
    ax.set_title(f"{run_id} temporal phi_mean")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    out_path = out_dir / f"{run_id}_temporal_phi_mean.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)


def _estimate_surface_z(
    phi_zyx: np.ndarray,
    *,
    valid_zyx: np.ndarray | None = None,
    band: float = 1.0,
) -> int:
    nz = int(phi_zyx.shape[0])
    densities: list[float] = []
    for z_idx in range(nz):
        plane = np.asarray(phi_zyx[z_idx], dtype=np.float32)
        if valid_zyx is not None:
            valid_plane = np.asarray(valid_zyx[z_idx], dtype=bool)
            denom = float(np.count_nonzero(valid_plane))
            if denom <= 0.0:
                densities.append(0.0)
                continue
            densities.append(float(np.count_nonzero((np.abs(plane) <= float(band)) & valid_plane) / denom))
            continue
        densities.append(float(np.mean(np.abs(plane) <= float(band))))
    return int(np.argmax(np.asarray(densities, dtype=np.float32)))


def _aperture_projection(phi_zyx: np.ndarray, valid_zyx: np.ndarray) -> np.ndarray:
    return np.any((phi_zyx <= 0.0) & valid_zyx, axis=0)


def _interface_depth(phi_zyx: np.ndarray, valid_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nz = int(phi_zyx.shape[0])
    z_index = np.arange(nz, dtype=np.int32).reshape(nz, 1, 1)
    open_mask = (phi_zyx <= 0.0) & valid_zyx
    has_open = np.any(open_mask, axis=0)
    first_depth = np.min(np.where(open_mask, z_index, nz), axis=0).astype(np.float32)
    first_depth = np.where(has_open, first_depth, np.nan)
    return first_depth, has_open


def _iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.count_nonzero(a & b))
    union = float(np.count_nonzero(a | b))
    if union <= 0.0:
        return 1.0
    return inter / union


def _signed_distance_2d(mask_xy: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_xy, dtype=bool)
    if _distance_transform_edt is not None:
        inside = _distance_transform_edt(mask)
        outside = _distance_transform_edt(~mask)
        return np.asarray(inside - outside, dtype=np.float32)
    return np.where(mask, 1.0, -1.0).astype(np.float32)


def _remove_border_connected(mask_xy: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_xy, dtype=bool)
    h, w = mask.shape
    if h == 0 or w == 0 or not np.any(mask):
        return mask
    visited = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(w):
        if mask[0, x]:
            stack.append((0, x))
        if mask[h - 1, x]:
            stack.append((h - 1, x))
    for y in range(h):
        if mask[y, 0]:
            stack.append((y, 0))
        if mask[y, w - 1]:
            stack.append((y, w - 1))
    while stack:
        y, x = stack.pop()
        if y < 0 or x < 0 or y >= h or x >= w:
            continue
        if visited[y, x] or (not mask[y, x]):
            continue
        visited[y, x] = True
        stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
    return mask & (~visited)


def _compute_shape_metrics(
    *,
    gt_np: Sequence[np.ndarray],
    pred_np: Sequence[np.ndarray],
    metric_mask_np: Sequence[np.ndarray],
) -> dict[str, float]:
    step_count = min(len(gt_np), len(pred_np), len(metric_mask_np))
    if step_count < 1:
        return {
            "top_aperture_iou_mean": 0.0,
            "top_aperture_iou_final": 0.0,
            "interface_depth_mae": 0.0,
            "aperture_boundary_leak_ratio": 0.0,
        }
    ious: list[float] = []
    depth_maes: list[float] = []
    leak_ratios: list[float] = []
    for idx in range(step_count):
        valid_mask = np.asarray(metric_mask_np[idx], dtype=bool)
        gt_ap = _aperture_projection(gt_np[idx], valid_mask)
        pd_ap = _aperture_projection(pred_np[idx], valid_mask)
        ious.append(_iou_bool(gt_ap, pd_ap))
        valid_inner_top = np.any(valid_mask, axis=0)
        pred_pixels = float(np.count_nonzero(pd_ap))
        leak_pixels = float(np.count_nonzero(pd_ap & (~valid_inner_top)))
        leak_ratios.append((leak_pixels / pred_pixels) if pred_pixels > 0.0 else 0.0)
        gt_depth, gt_has = _interface_depth(gt_np[idx], valid_mask)
        pd_depth, pd_has = _interface_depth(pred_np[idx], valid_mask)
        use = gt_has | pd_has
        if np.any(use):
            diff = np.abs(pd_depth[use] - gt_depth[use])
            finite = diff[np.isfinite(diff)]
            if finite.size > 0:
                depth_maes.append(float(np.mean(finite)))
    return {
        "top_aperture_iou_mean": float(np.mean(ious)),
        "top_aperture_iou_final": float(ious[-1]),
        "interface_depth_mae": float(np.mean(depth_maes)) if depth_maes else 0.0,
        "aperture_boundary_leak_ratio": float(np.mean(leak_ratios)) if leak_ratios else 0.0,
    }


def _render_topview_views(
    *,
    gt_frames: list[Any],
    pred_frames: list[Any],
    plot_masks: list[np.ndarray] | None,
    target_material_masks: list[np.ndarray] | None,
    out_dir: Path,
    run_id: str,
    visual_band: float,
    target_material_id: int,
    phi_boundary_clip_vox: int,
    analysis_xy_margin_vox: int,
    analysis_z_margin_vox: int,
    dpi: int,
) -> dict[str, Any]:
    plt = load_pyplot()
    if plt is None:
        return {"enabled": False, "reason": "matplotlib unavailable"}

    sdf_dir = out_dir / "top_xy_sdf"
    aperture_dir = out_dir / "top_xy_aperture"
    material_dir = out_dir / "top_xy_material_overlay"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    aperture_dir.mkdir(parents=True, exist_ok=True)
    material_dir.mkdir(parents=True, exist_ok=True)

    gt_np = [_to_numpy_zyx(frame) for frame in gt_frames]
    pred_np = [_to_numpy_zyx(frame) for frame in pred_frames]
    step_count = min(len(gt_np), len(pred_np))
    if step_count < 1:
        return {"enabled": False, "reason": "no comparable frames"}
    if plot_masks is None or len(plot_masks) < step_count:
        valid_np = [np.ones_like(gt_np[idx], dtype=bool) for idx in range(step_count)]
    else:
        valid_np = [np.asarray(plot_masks[idx], dtype=bool) for idx in range(step_count)]
    if target_material_masks is None or len(target_material_masks) < step_count:
        target_np = [np.zeros_like(gt_np[idx], dtype=bool) for idx in range(step_count)]
    else:
        target_np = [np.asarray(target_material_masks[idx], dtype=bool) & valid_np[idx] for idx in range(step_count)]

    # Focus ROI around target-hole region to avoid domain-border-dominant plots.
    target_union_top = np.zeros_like(np.any(valid_np[0], axis=0), dtype=bool)
    for idx in range(step_count):
        target_union_top |= np.any(target_np[idx], axis=0)
    if np.any(target_union_top):
        ys, xs = np.where(target_union_top)
    else:
        fallback_top = np.any(valid_np[0], axis=0)
        ys, xs = np.where(fallback_top)
    if ys.size > 0 and xs.size > 0:
        crop_margin = 6
        y0 = max(int(np.min(ys)) - crop_margin, 0)
        y1 = min(int(np.max(ys)) + crop_margin, int(target_union_top.shape[0]) - 1)
        x0 = max(int(np.min(xs)) - crop_margin, 0)
        x1 = min(int(np.max(xs)) + crop_margin, int(target_union_top.shape[1]) - 1)
    else:
        y0, y1 = 0, int(target_union_top.shape[0]) - 1
        x0, x1 = 0, int(target_union_top.shape[1]) - 1

    gt_ap_full = [_remove_border_connected(_aperture_projection(gt_np[idx], valid_np[idx])) for idx in range(step_count)]
    pred_ap_full = [_remove_border_connected(_aperture_projection(pred_np[idx], valid_np[idx])) for idx in range(step_count)]
    gt_slices = [_signed_distance_2d(gt_ap_full[idx])[y0 : y1 + 1, x0 : x1 + 1] for idx in range(step_count)]
    pred_slices = [_signed_distance_2d(pred_ap_full[idx])[y0 : y1 + 1, x0 : x1 + 1] for idx in range(step_count)]
    valid_slices = [np.ones_like(gt_slices[idx], dtype=bool) for idx in range(step_count)]
    err_slices = [np.abs(pred_slices[idx] - gt_slices[idx]) for idx in range(step_count)]

    vmax = float(max(1.0, visual_band))
    vmin = -vmax
    err_max = float(np.percentile(np.concatenate([arr.ravel() for arr in err_slices]), 99.0))
    err_max = max(err_max, 1e-6)
    cmap_sdf = plt.get_cmap("coolwarm").copy()
    cmap_err = plt.get_cmap("magma").copy()

    for idx in range(step_count):
        gt_slice = np.ma.array(gt_slices[idx], mask=~valid_slices[idx])
        pred_slice = np.ma.array(pred_slices[idx], mask=~valid_slices[idx])
        err_slice = np.ma.array(err_slices[idx], mask=~valid_slices[idx])

        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
        for ax, arr, title in (
            (axes[0], gt_slice, "GT SDF"),
            (axes[1], pred_slice, "PRED SDF"),
            (axes[2], err_slice, "|PRED-GT|"),
        ):
            if title == "|PRED-GT|":
                image = ax.imshow(arr, cmap=cmap_err, origin="lower", vmin=0.0, vmax=err_max)
            else:
                image = ax.imshow(arr, cmap=cmap_sdf, origin="lower", vmin=vmin, vmax=vmax)
                contour_data = np.asarray(arr)
                ax.contour(contour_data, levels=[0.0], colors="black", linewidths=0.8)
            ax.set_title(f"{title} (top-aperture SDF)")
            ax.set_xlabel(f"x index [{x0}:{x1}]")
            ax.set_ylabel(f"y index [{y0}:{y1}]")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(sdf_dir / f"top_xy_sdf_t{idx:04d}.png", dpi=max(72, int(dpi)))
        plt.close(fig)

        gt_ap = gt_ap_full[idx][y0 : y1 + 1, x0 : x1 + 1]
        pd_ap = pred_ap_full[idx][y0 : y1 + 1, x0 : x1 + 1]
        ap_err = np.logical_xor(gt_ap, pd_ap)
        fig2, axes2 = plt.subplots(1, 3, figsize=(13.2, 4.2))
        for ax, arr, title in (
            (axes2[0], gt_ap, "GT aperture"),
            (axes2[1], pd_ap, "PRED aperture"),
            (axes2[2], ap_err, "Aperture XOR"),
        ):
            image = ax.imshow(arr.astype(np.float32), cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
            ax.set_title(title)
            ax.set_xlabel("x index")
            ax.set_ylabel("y index")
            fig2.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig2.tight_layout()
        fig2.savefig(aperture_dir / f"top_xy_aperture_t{idx:04d}.png", dpi=max(72, int(dpi)))
        plt.close(fig2)

        # 0: outside valid, 1: valid non-target, 2: target material
        valid_top = np.any(valid_np[idx], axis=0)[y0 : y1 + 1, x0 : x1 + 1]
        target_top = np.any(target_np[idx], axis=0)[y0 : y1 + 1, x0 : x1 + 1]
        cls = np.zeros_like(valid_top, dtype=np.int32)
        cls[valid_top] = 1
        cls[target_top] = 2
        fig3 = plt.figure(figsize=(4.8, 4.2))
        ax3 = fig3.add_subplot(1, 1, 1)
        cmap = plt.get_cmap("tab20c")
        image3 = ax3.imshow(cls.astype(np.float32), cmap=cmap, origin="lower", vmin=0.0, vmax=2.0)
        ax3.set_title(f"Material overlay (target={target_material_id})")
        ax3.set_xlabel("x index")
        ax3.set_ylabel("y index")
        cbar3 = fig3.colorbar(image3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_ticks([0.0, 1.0, 2.0])
        cbar3.set_ticklabels(["outside-valid", "valid non-target", f"target {target_material_id}"])
        fig3.tight_layout()
        fig3.savefig(material_dir / f"top_xy_material_overlay_t{idx:04d}.png", dpi=max(72, int(dpi)))
        plt.close(fig3)

    manifest = {
        "run_id": str(run_id),
        "num_frames": int(step_count),
        "view": "top_xy",
        "sdf_view_mode": "top_aperture_signed_distance",
        "top_xy_sdf_dir": str(sdf_dir.resolve()),
        "top_xy_aperture_dir": str(aperture_dir.resolve()),
        "top_xy_material_overlay_dir": str(material_dir.resolve()),
        "top_xy_sdf_png_count": int(step_count),
        "top_xy_aperture_png_count": int(step_count),
        "top_xy_material_overlay_png_count": int(step_count),
        "focus_crop_bbox_xy": {"x_min": int(x0), "x_max": int(x1), "y_min": int(y0), "y_max": int(y1)},
        "domain_mask_applied": True,
        "material_overlay_enabled": True,
        "target_material_id": int(target_material_id),
        "phi_boundary_clip_vox": int(phi_boundary_clip_vox),
        "analysis_xy_margin_vox": int(analysis_xy_margin_vox),
        "analysis_z_margin_vox": int(analysis_z_margin_vox),
        "plot_mask_policy": "valid_inner",
        "outer_boundary_policy": "remove_border_connected_components",
        "aperture_definition": "top projection where any(phi<=0) over z within ValidMask",
        "colormap": {
            "gt_pred": {
                "name": "coolwarm",
                "vmin": float(vmin),
                "vmax": float(vmax),
                "zero_level_contour": True,
                "meaning": "signed distance on top-aperture mask: 0=hole boundary, >0 hole interior, <0 exterior",
            },
            "sdf_abs_error": {
                "name": "magma",
                "vmin": 0.0,
                "vmax": float(err_max),
                "meaning": "absolute signed-distance error on top XY slice",
            },
        },
    }
    with (out_dir / "images_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    return {"enabled": True, "manifest_path": str((out_dir / "images_manifest.json"))}


def _orthogonal_indices(valid_zyx: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(np.asarray(valid_zyx, dtype=bool))
    if coords.size == 0:
        return 0, max(0, valid_zyx.shape[1] // 2), max(0, valid_zyx.shape[2] // 2)
    z_idx = int(round(float(np.mean(coords[:, 0]))))
    y_idx = int(round(float(np.mean(coords[:, 1]))))
    x_idx = int(round(float(np.mean(coords[:, 2]))))
    z_idx = max(0, min(z_idx, int(valid_zyx.shape[0]) - 1))
    y_idx = max(0, min(y_idx, int(valid_zyx.shape[1]) - 1))
    x_idx = max(0, min(x_idx, int(valid_zyx.shape[2]) - 1))
    return z_idx, y_idx, x_idx


def _render_orthogonal_sdf_views(
    *,
    gt_frames: list[Any],
    pred_frames: list[Any],
    valid_masks: list[np.ndarray],
    out_dir: Path,
    run_id: str,
    visual_band: float,
    render_xy: bool,
    render_xz: bool,
    render_yz: bool,
    render_error: bool,
    dpi: int,
) -> dict[str, Any]:
    plt = load_pyplot()
    if plt is None:
        return {"enabled": False, "reason": "matplotlib unavailable"}
    gt_np = [_to_numpy_zyx(frame) for frame in gt_frames]
    pred_np = [_to_numpy_zyx(frame) for frame in pred_frames]
    step_count = min(len(gt_np), len(pred_np), len(valid_masks))
    if step_count <= 0:
        return {"enabled": False, "reason": "no frames"}

    view_flags = {"xy": bool(render_xy), "xz": bool(render_xz), "yz": bool(render_yz)}
    base_dir = out_dir
    for view, enabled in view_flags.items():
        if enabled:
            (base_dir / view).mkdir(parents=True, exist_ok=True)

    vmax = float(max(1.0, visual_band))
    vmin = -vmax
    frame_count = 0
    for idx in range(step_count):
        gt = gt_np[idx]
        pred = pred_np[idx]
        valid = np.asarray(valid_masks[idx], dtype=bool)
        z_idx, y_idx, x_idx = _orthogonal_indices(valid)
        planes = {
            "xy": (gt[z_idx], pred[z_idx], valid[z_idx], f"z={z_idx}"),
            "xz": (gt[:, y_idx, :], pred[:, y_idx, :], valid[:, y_idx, :], f"y={y_idx}"),
            "yz": (gt[:, :, x_idx], pred[:, :, x_idx], valid[:, :, x_idx], f"x={x_idx}"),
        }
        for view, enabled in view_flags.items():
            if not enabled:
                continue
            gt_plane, pred_plane, valid_plane, axis_note = planes[view]
            gt_masked = np.ma.array(gt_plane, mask=~valid_plane)
            pred_masked = np.ma.array(pred_plane, mask=~valid_plane)
            err_masked = np.ma.array(np.abs(pred_plane - gt_plane), mask=~valid_plane)
            cols = 3 if render_error else 2
            fig, axes = plt.subplots(1, cols, figsize=(4.4 * cols, 3.8))
            if cols == 1:
                axes = [axes]
            elif hasattr(axes, "tolist"):
                axes = axes.tolist()
            panels = [
                ("GT", gt_masked, "coolwarm", vmin, vmax),
                ("PRED", pred_masked, "coolwarm", vmin, vmax),
            ]
            if render_error:
                err_max = float(np.percentile(err_masked.compressed(), 99.0)) if err_masked.count() > 0 else 1.0
                panels.append(("ABS_ERR", err_masked, "magma", 0.0, max(err_max, 1e-6)))
            for axis, (title, arr, cmap, lo, hi) in zip(axes, panels):
                im = axis.imshow(arr, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
                if title != "ABS_ERR":
                    contour_data = np.asarray(arr)
                    if contour_data.ndim == 2 and contour_data.shape[0] >= 2 and contour_data.shape[1] >= 2:
                        axis.contour(contour_data, levels=[0.0], colors="black", linewidths=0.7)
                axis.set_title(f"{title} {view.upper()} ({axis_note})")
                axis.set_xlabel("u")
                axis.set_ylabel("v")
                fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(base_dir / view / f"{run_id}_{view}_sdf_t{idx:04d}.png", dpi=max(72, int(dpi)))
            plt.close(fig)
        frame_count += 1

    return {
        "enabled": True,
        "views": view_flags,
        "frame_count": int(frame_count),
        "render_error": bool(render_error),
        "xy_dir": str((base_dir / "xy").resolve()) if view_flags["xy"] else None,
        "xz_dir": str((base_dir / "xz").resolve()) if view_flags["xz"] else None,
        "yz_dir": str((base_dir / "yz").resolve()) if view_flags["yz"] else None,
    }


def _compute_rollout_metrics_local(
    *,
    predicted_runs: list[list[Any]],
    reference_runs: list[SyntheticSDFRun],
) -> dict[str, float]:
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    for run_idx, ref_run in enumerate(reference_runs):
        if run_idx >= len(predicted_runs):
            continue
        pred_seq = predicted_runs[run_idx]
        ref_seq = ref_run.phi_t
        step_count = min(len(pred_seq), len(ref_seq))
        for step_idx in range(step_count):
            pred_vals = _flatten_values(pred_seq[step_idx])
            ref_vals = _flatten_values(ref_seq[step_idx])
            n = min(len(pred_vals), len(ref_vals))
            for i in range(n):
                diff = float(pred_vals[i]) - float(ref_vals[i])
                abs_errors.append(abs(diff))
                sq_errors.append(diff * diff)
    if not abs_errors:
        return {"sdf_l1_mean": 0.0, "sdf_l2_rmse": 0.0}
    return {
        "sdf_l1_mean": float(sum(abs_errors) / float(len(abs_errors))),
        "sdf_l2_rmse": float((sum(sq_errors) / float(len(sq_errors))) ** 0.5),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval-dataset-3d-test2-holdout",
        description="Evaluate holdout rollout metrics for ataset_3d_test2 pilot runs.",
    )
    parser.add_argument("--model-state", required=True, help="Path to train/outputs/model_state.json")
    parser.add_argument("--holdout-json", required=True, help="Path to prepared holdout_dataset.json")
    parser.add_argument("--out-dir", required=True, help="Output directory for holdout metrics.")
    parser.add_argument(
        "--split-manifest",
        default="",
        help="Optional path to split_manifest.json (defaults to sibling of holdout-json).",
    )
    parser.add_argument(
        "--valid-mask-array",
        default="ValidMask",
        help="VTI array name used as domain-valid mask for visualization/shape metrics.",
    )
    parser.add_argument(
        "--phi-boundary-clip-vox",
        type=int,
        default=-1,
        help="Override phi boundary clip vox. Default -1 uses split_manifest value.",
    )
    parser.add_argument(
        "--analysis-xy-margin-vox",
        type=int,
        default=2,
        help="Additional XY margin (vox) removed from metric mask for pilot-only boundary suppression.",
    )
    parser.add_argument(
        "--analysis-z-margin-vox",
        type=int,
        default=1,
        help="Additional Z margin (vox) removed from metric mask for pilot-only boundary suppression.",
    )
    parser.add_argument(
        "--viz-config-yaml",
        default="",
        help="Optional visualization profile YAML. Priority: CLI > stage/run defaults.",
    )
    parser.add_argument("--reinit-enabled", action="store_true", help="Enable Sussman reinit during rollout.")
    parser.add_argument("--reinit-every-n", type=int, default=5, help="Reinitialize every N steps.")
    parser.add_argument("--reinit-iters", type=int, default=8, help="Sussman reinit iterations.")
    parser.add_argument("--reinit-dt", type=float, default=0.3, help="Sussman reinit dt.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_state_path = Path(args.model_state)
    holdout_json_path = Path(args.holdout_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_state_path.exists() or not model_state_path.is_file():
        raise ValueError(f"model-state does not exist: {model_state_path}")
    if not holdout_json_path.exists() or not holdout_json_path.is_file():
        raise ValueError(f"holdout-json does not exist: {holdout_json_path}")

    model, model_state = _load_model_from_state(model_state_path)
    dataset = _load_dataset(holdout_json_path)
    split_manifest = _load_split_manifest(holdout_json_path)
    if str(args.split_manifest).strip():
        with Path(args.split_manifest).open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if isinstance(payload, Mapping):
            split_manifest = {str(k): v for k, v in payload.items()}
    frame_path_map = _frame_paths_by_run(split_manifest)
    visual_band = float(max(1.0, 2.0 * float(split_manifest.get("band_width", 0.5))))
    target_material_id = int(split_manifest.get("selected_target_material_id", 10))
    viz_warnings: list[str] = []
    viz_cfg = resolve_visualization_config(
        cli_path=args.viz_config_yaml if str(args.viz_config_yaml).strip() else None,
        run_config=split_manifest.get("visualization") if isinstance(split_manifest.get("visualization"), Mapping) else {},
        warnings=viz_warnings,
    )
    export_cfg = viz_cfg.get("export")
    dpi = int(export_cfg.get("dpi", 140)) if isinstance(export_cfg, Mapping) else 140
    sdf_xy_enabled = viz_enabled(viz_cfg, "sdf_views.xy", True)
    sdf_xz_enabled = viz_enabled(viz_cfg, "sdf_views.xz", True)
    sdf_yz_enabled = viz_enabled(viz_cfg, "sdf_views.yz", True)
    sdf_err_enabled = viz_enabled(viz_cfg, "sdf_views.gt_pred_error", True)
    if int(args.phi_boundary_clip_vox) >= 0:
        phi_boundary_clip_vox = int(args.phi_boundary_clip_vox)
    else:
        phi_boundary_clip_vox = int(split_manifest.get("phi_boundary_clip_vox", 3))

    sim_opts: dict[str, Any] = {
        "reinit_enabled": bool(args.reinit_enabled),
        "reinit_every_n": int(max(1, args.reinit_every_n)),
        "reinit_iters": int(max(1, args.reinit_iters)),
        "reinit_dt": float(args.reinit_dt),
    }

    predicted_runs: list[list[Any]] = []
    per_run_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    viz_records: list[dict[str, Any]] = []
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for run in dataset.runs:
        pred = rollout(run, model, simulation_options=sim_opts)
        predicted_runs.append(pred)
        gt_np = [_to_numpy_zyx(frame) for frame in run.phi_t]
        frame_paths = frame_path_map.get(str(run.run_id), [])
        if frame_paths:
            valid_np = _load_valid_masks_from_paths(
                frame_paths=frame_paths[: len(gt_np)],
                valid_mask_array=str(args.valid_mask_array),
                fallback_shape=gt_np[0].shape,
            )
            target_material_np = _load_material_masks_from_paths(
                frame_paths=frame_paths[: len(gt_np)],
                target_material_id=int(target_material_id),
                fallback_shape=gt_np[0].shape,
            )
            if len(valid_np) < len(gt_np):
                valid_np.extend([np.ones_like(gt_np[0], dtype=bool) for _ in range(len(gt_np) - len(valid_np))])
            if len(target_material_np) < len(gt_np):
                target_material_np.extend([np.zeros_like(gt_np[0], dtype=bool) for _ in range(len(gt_np) - len(target_material_np))])
        else:
            valid_np = [np.ones_like(gt_np[idx], dtype=bool) for idx in range(len(gt_np))]
            target_material_np = [np.zeros_like(gt_np[idx], dtype=bool) for idx in range(len(gt_np))]
        valid_inner_np = [_erode_mask(valid_np[idx], int(max(0, phi_boundary_clip_vox))) for idx in range(len(gt_np))]
        metric_mask_np = [
            _shrink_analysis_mask(
                valid_inner_np[idx],
                xy_margin_vox=int(max(0, args.analysis_xy_margin_vox)),
                z_margin_vox=int(max(0, args.analysis_z_margin_vox)),
            )
            for idx in range(len(gt_np))
        ]
        # Guardrail: if mask becomes too small, fallback to valid_inner for metrics.
        for idx in range(len(metric_mask_np)):
            inner_count = int(np.count_nonzero(valid_inner_np[idx]))
            metric_count = int(np.count_nonzero(metric_mask_np[idx]))
            if inner_count > 0 and (float(metric_count) / float(inner_count)) < 0.05:
                metric_mask_np[idx] = np.asarray(valid_inner_np[idx], dtype=bool)
        pred_np = [_to_numpy_zyx(frame) for frame in pred]
        shape_metrics = _compute_shape_metrics(
            gt_np=gt_np,
            pred_np=pred_np,
            metric_mask_np=metric_mask_np,
        )
        temporal = compute_temporal_diagnostics(
            predicted_phi_t=pred,
            reference_phi_t=run.phi_t,
        )
        plot_path = _write_temporal_plot(
            run_id=str(run.run_id),
            pred_phi_t=pred,
            ref_phi_t=run.phi_t,
            out_dir=image_dir,
        )
        row = {
            "run_id": str(run.run_id),
            "steps": int(min(len(pred), len(run.phi_t))),
            "final_frame_l1": _final_frame_l1(pred[-1], run.phi_t[-1]),
            "vn_frame_mean_mae": _vn_frame_mean_mae(pred, run.phi_t, float(run.dt)),
            "delta_phi_sign_agreement": float(temporal.get("delta_phi_sign_agreement", 0.0)),
            "early_window_error": float(temporal.get("early_window_error", 0.0)),
            "late_window_error": float(temporal.get("late_window_error", 0.0)),
            "r2_all_frames": float(temporal.get("r2_all_frames", 0.0)),
            "r2_final_frame": float(temporal.get("r2_final_frame", 0.0)),
            "top_aperture_iou_mean": float(shape_metrics.get("top_aperture_iou_mean", 0.0)),
            "top_aperture_iou_final": float(shape_metrics.get("top_aperture_iou_final", 0.0)),
            "interface_depth_mae": float(shape_metrics.get("interface_depth_mae", 0.0)),
            "aperture_boundary_leak_ratio": float(shape_metrics.get("aperture_boundary_leak_ratio", 0.0)),
            "temporal_plot_path": plot_path,
        }
        per_run_rows.append(row)
        temporal_rows.append(
            {
                "run_id": str(run.run_id),
                **temporal,
                "temporal_plot_path": plot_path,
            }
        )
        if sdf_xy_enabled:
            top_manifest = _render_topview_views(
                gt_frames=run.phi_t,
                pred_frames=pred,
                plot_masks=valid_np,
                target_material_masks=target_material_np,
                out_dir=image_dir,
                run_id=str(run.run_id),
                visual_band=float(visual_band),
                target_material_id=int(target_material_id),
                phi_boundary_clip_vox=int(phi_boundary_clip_vox),
                analysis_xy_margin_vox=int(max(0, args.analysis_xy_margin_vox)),
                analysis_z_margin_vox=int(max(0, args.analysis_z_margin_vox)),
                dpi=int(dpi),
            )
        else:
            top_manifest = {"enabled": False, "reason": "disabled by visualization config"}
        orth_manifest = _render_orthogonal_sdf_views(
            gt_frames=run.phi_t,
            pred_frames=pred,
            valid_masks=valid_np,
            out_dir=image_dir,
            run_id=str(run.run_id),
            visual_band=float(visual_band),
            render_xy=bool(sdf_xy_enabled),
            render_xz=bool(sdf_xz_enabled),
            render_yz=bool(sdf_yz_enabled),
            render_error=bool(sdf_err_enabled),
            dpi=int(dpi),
        )
        viz_records.append(
            {
                "run_id": str(run.run_id),
                "top_xy": top_manifest,
                "orthogonal_sdf": orth_manifest,
            }
        )

    agg_metrics = _compute_rollout_metrics_local(predicted_runs=predicted_runs, reference_runs=dataset.runs)
    final_frame_l1_vals = [float(row["final_frame_l1"]) for row in per_run_rows]
    vn_frame_mean_mae_vals = [float(row["vn_frame_mean_mae"]) for row in per_run_rows]
    delta_sign_vals = [float(row["delta_phi_sign_agreement"]) for row in per_run_rows]
    early_vals = [float(row["early_window_error"]) for row in per_run_rows]
    late_vals = [float(row["late_window_error"]) for row in per_run_rows]
    r2_all_vals = [float(row["r2_all_frames"]) for row in per_run_rows]
    r2_final_vals = [float(row["r2_final_frame"]) for row in per_run_rows]
    aperture_iou_vals = [float(row["top_aperture_iou_mean"]) for row in per_run_rows]
    aperture_iou_final_vals = [float(row["top_aperture_iou_final"]) for row in per_run_rows]
    depth_mae_vals = [float(row["interface_depth_mae"]) for row in per_run_rows]
    leak_vals = [float(row["aperture_boundary_leak_ratio"]) for row in per_run_rows]
    summary = {
        "sdf_l1_mean": float(agg_metrics.get("sdf_l1_mean", 0.0)),
        "sdf_l2_rmse": float(agg_metrics.get("sdf_l2_rmse", 0.0)),
        "final_frame_l1": float(fmean(final_frame_l1_vals)) if final_frame_l1_vals else 0.0,
        "vn_frame_mean_mae": float(fmean(vn_frame_mean_mae_vals)) if vn_frame_mean_mae_vals else 0.0,
        "delta_phi_sign_agreement": float(fmean(delta_sign_vals)) if delta_sign_vals else 0.0,
        "early_window_error": float(fmean(early_vals)) if early_vals else 0.0,
        "late_window_error": float(fmean(late_vals)) if late_vals else 0.0,
        "r2_all_frames": float(fmean(r2_all_vals)) if r2_all_vals else 0.0,
        "r2_final_frame": float(fmean(r2_final_vals)) if r2_final_vals else 0.0,
        "top_aperture_iou_mean": float(fmean(aperture_iou_vals)) if aperture_iou_vals else 0.0,
        "top_aperture_iou_final": float(fmean(aperture_iou_final_vals)) if aperture_iou_final_vals else 0.0,
        "interface_depth_mae": float(fmean(depth_mae_vals)) if depth_mae_vals else 0.0,
        "aperture_boundary_leak_ratio": float(fmean(leak_vals)) if leak_vals else 0.0,
        "num_runs_evaluated": len(per_run_rows),
    }

    payload = {
        "summary": summary,
        "per_run": per_run_rows,
        "split_manifest": split_manifest if split_manifest else None,
        "domain_policy": {
            "target_material_id": int(target_material_id),
            "phi_boundary_clip_vox": int(phi_boundary_clip_vox),
            "valid_mask_array": str(args.valid_mask_array),
            "analysis_xy_margin_vox": int(max(0, args.analysis_xy_margin_vox)),
            "analysis_z_margin_vox": int(max(0, args.analysis_z_margin_vox)),
            "plot_mask_policy": "valid_inner",
        },
        "model_state": {
            "model_backend": str(model_state.get("model_backend", "")),
            "model_name": str(model_state.get("model_name", "")),
            "checkpoint_path": str(model_state.get("checkpoint_path", "")),
        },
        "runtime_capabilities": detect_runtime_capabilities().missing_summary(),
        "simulation_options": sim_opts,
    }
    out_json = out_dir / "holdout_eval.json"
    out_csv = out_dir / "holdout_eval.csv"
    temporal_json = out_dir / "temporal_diagnostics.json"
    with out_json.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    with temporal_json.open("w", encoding="utf-8") as fp:
        json.dump({"summary": summary, "records": temporal_rows}, fp, indent=2)

    csv_rows = [
        {"metric": "sdf_l1_mean", "value": summary["sdf_l1_mean"]},
        {"metric": "sdf_l2_rmse", "value": summary["sdf_l2_rmse"]},
        {"metric": "final_frame_l1", "value": summary["final_frame_l1"]},
        {"metric": "vn_frame_mean_mae", "value": summary["vn_frame_mean_mae"]},
        {"metric": "delta_phi_sign_agreement", "value": summary["delta_phi_sign_agreement"]},
        {"metric": "early_window_error", "value": summary["early_window_error"]},
        {"metric": "late_window_error", "value": summary["late_window_error"]},
        {"metric": "r2_all_frames", "value": summary["r2_all_frames"]},
        {"metric": "r2_final_frame", "value": summary["r2_final_frame"]},
        {"metric": "top_aperture_iou_mean", "value": summary["top_aperture_iou_mean"]},
        {"metric": "top_aperture_iou_final", "value": summary["top_aperture_iou_final"]},
        {"metric": "interface_depth_mae", "value": summary["interface_depth_mae"]},
        {"metric": "aperture_boundary_leak_ratio", "value": summary["aperture_boundary_leak_ratio"]},
        {"metric": "num_runs_evaluated", "value": summary["num_runs_evaluated"]},
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(csv_rows)

    viz_manifest_path = write_visualization_manifest(
        out_dir / "visualization_manifest.json",
        {
            "config": viz_cfg,
            "warnings": viz_warnings,
            "records": viz_records,
            "paths": {
                "out_json": str(out_json),
                "out_csv": str(out_csv),
                "temporal_json": str(temporal_json),
                "images_dir": str(image_dir),
            },
        },
    )
    report_index_path = out_dir / "report_index.json"
    report_payload = {
        "summary_json": str(out_json),
        "summary_csv": str(out_csv),
        "temporal_json": str(temporal_json),
        "visualization_manifest_json": str(viz_manifest_path),
        "images_dir": str(image_dir),
        "key_metrics": summary,
    }
    with report_index_path.open("w", encoding="utf-8") as fp:
        json.dump(report_payload, fp, indent=2)

    print(f"holdout eval json: {out_json}")
    print(f"holdout eval csv: {out_csv}")
    print(f"temporal diagnostics json: {temporal_json}")
    print(f"temporal plots dir: {image_dir}")
    print(f"visualization manifest: {viz_manifest_path}")
    print(f"report index: {report_index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
