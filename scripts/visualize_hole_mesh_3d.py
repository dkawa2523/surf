#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_dataset_3d_test2_holdout as eval_holdout
from wafer_surrogate.core.rollout import rollout
from wafer_surrogate.viz.utils import resolve_visualization_config, viz_enabled, write_visualization_manifest


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray  # (N,3) float32
    faces: np.ndarray  # (M,3) int32
    voxel_count: int


def _apply_quarter_cutaway(mask_zyx: np.ndarray, *, center: tuple[float, float, float] | None) -> np.ndarray:
    mask = np.asarray(mask_zyx, dtype=bool).copy()
    if not np.any(mask):
        return mask
    if center is None:
        c = _component_center(mask)
        if c is None:
            return mask
        cz, cy, cx = c
    else:
        cz, cy, cx = center
    y_cut = int(round(cy))
    x_cut = int(round(cx))
    y_cut = max(0, min(y_cut, mask.shape[1] - 1))
    x_cut = max(0, min(x_cut, mask.shape[2] - 1))
    # Remove one XY quarter across full depth to expose interior walls.
    mask[:, y_cut:, x_cut:] = False
    return mask


def _parse_frames(raw: str, max_len: int) -> list[int]:
    if str(raw).strip().lower() == "all":
        return list(range(max_len))
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx = max_len + idx
        if 0 <= idx < max_len:
            out.append(idx)
    uniq = sorted(set(out))
    if not uniq:
        raise ValueError(f"no valid frame indices from --frames={raw}")
    return uniq


def _component_center(mask: np.ndarray) -> tuple[float, float, float] | None:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    c = np.mean(idx, axis=0)
    return float(c[0]), float(c[1]), float(c[2])


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return []
    labels, n = ndimage.label(binary)
    out: list[np.ndarray] = []
    for label_id in range(1, int(n) + 1):
        cm = labels == label_id
        if np.any(cm):
            out.append(cm)
    return out


def _touches_border(mask: np.ndarray) -> bool:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return False
    z_min, y_min, x_min = np.min(idx, axis=0)
    z_max, y_max, x_max = np.max(idx, axis=0)
    shape = mask.shape
    return bool(
        z_min == 0
        or y_min == 0
        or x_min == 0
        or z_max == (shape[0] - 1)
        or y_max == (shape[1] - 1)
        or x_max == (shape[2] - 1)
    )


def _select_focus_component(mask: np.ndarray, reference_center: tuple[float, float, float] | None = None) -> np.ndarray:
    comps = _component_masks(mask)
    if not comps:
        return np.zeros_like(mask, dtype=bool)
    valid = [cm for cm in comps if not _touches_border(cm)]
    candidates = valid if valid else comps
    if reference_center is None:
        candidates.sort(key=lambda cm: -int(np.count_nonzero(cm)))
        return candidates[0]

    rz, ry, rx = reference_center
    best: np.ndarray | None = None
    best_score: tuple[float, float] | None = None
    for cm in candidates:
        c = _component_center(cm)
        if c is None:
            continue
        dz, dy, dx = c[0] - rz, c[1] - ry, c[2] - rx
        dist = float(np.sqrt(dz * dz + dy * dy + dx * dx))
        area = int(np.count_nonzero(cm))
        score = (dist, -float(area))
        if best_score is None or score < best_score:
            best_score = score
            best = cm
    if best is not None:
        return best
    candidates.sort(key=lambda cm: -int(np.count_nonzero(cm)))
    return candidates[0]


def _extract_hole_mask(phi_zyx: np.ndarray, valid_zyx: np.ndarray, ref_center: tuple[float, float, float] | None = None) -> np.ndarray:
    inside = (np.asarray(phi_zyx, dtype=np.float32) > 0.0) & np.asarray(valid_zyx, dtype=bool)
    return _select_focus_component(inside, reference_center=ref_center)


def _mask_to_surface_mesh(mask_zyx: np.ndarray) -> MeshData:
    mask = np.asarray(mask_zyx, dtype=bool)
    if not np.any(mask):
        return MeshData(vertices=np.zeros((0, 3), dtype=np.float32), faces=np.zeros((0, 3), dtype=np.int32), voxel_count=0)

    boundary = mask & (~ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool)))
    voxels = np.argwhere(boundary)
    if voxels.size == 0:
        return MeshData(vertices=np.zeros((0, 3), dtype=np.float32), faces=np.zeros((0, 3), dtype=np.int32), voxel_count=0)

    directions = (
        (-1, 0, 0),  # z-
        (1, 0, 0),  # z+
        (0, -1, 0),  # y-
        (0, 1, 0),  # y+
        (0, 0, -1),  # x-
        (0, 0, 1),  # x+
    )
    # Vertex order in XYZ coordinates (x, y, z) for unit voxel.
    face_quads = {
        (-1, 0, 0): ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
        (1, 0, 0): ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
        (0, -1, 0): ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
        (0, 1, 0): ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
        (0, 0, -1): ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
        (0, 0, 1): ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
    }

    v_map: dict[tuple[float, float, float], int] = {}
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []

    z_max, y_max, x_max = mask.shape

    def v_idx(v: tuple[float, float, float]) -> int:
        i = v_map.get(v)
        if i is not None:
            return i
        i = len(verts)
        v_map[v] = i
        verts.append(v)
        return i

    for z, y, x in voxels:
        for dz, dy, dx in directions:
            nz, ny, nx = int(z + dz), int(y + dy), int(x + dx)
            if 0 <= nz < z_max and 0 <= ny < y_max and 0 <= nx < x_max and mask[nz, ny, nx]:
                continue
            quad = face_quads[(dz, dy, dx)]
            pts = [(float(x + qx), float(y + qy), float(z + qz)) for (qx, qy, qz) in quad]
            i0, i1, i2, i3 = (v_idx(pts[0]), v_idx(pts[1]), v_idx(pts[2]), v_idx(pts[3]))
            tris.append((i0, i1, i2))
            tris.append((i0, i2, i3))

    return MeshData(
        vertices=np.asarray(verts, dtype=np.float32),
        faces=np.asarray(tris, dtype=np.int32),
        voxel_count=int(np.count_nonzero(mask)),
    )


def _write_ply(path: Path, mesh: MeshData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {mesh.vertices.shape[0]}\n")
        fp.write("property float x\nproperty float y\nproperty float z\n")
        fp.write(f"element face {mesh.faces.shape[0]}\n")
        fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")
        for x, y, z in mesh.vertices:
            fp.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for i, j, k in mesh.faces:
            fp.write(f"3 {int(i)} {int(j)} {int(k)}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build 3D hole-shape meshes from holdout GT/PRED SDF and export interactive HTML.")
    p.add_argument("--model-state", required=True)
    p.add_argument("--holdout-json", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", default="")
    p.add_argument("--frames", default="0,4,8", help="Comma-separated indices or 'all'.")
    p.add_argument("--voxel-stride", type=int, default=1, help="Downsample factor for mesh extraction (1=full).")
    cutaway_group = p.add_mutually_exclusive_group()
    cutaway_group.add_argument("--enable-cutaway", dest="enable_cutaway", action="store_true", help="Include cutaway row to reveal interior hole walls.")
    cutaway_group.add_argument("--disable-cutaway", dest="enable_cutaway", action="store_false", help="Disable cutaway row rendering.")
    p.set_defaults(enable_cutaway=None)
    p.add_argument("--viz-config-yaml", default="", help="Optional visualization profile YAML.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_warnings: list[str] = []
    viz_cfg = resolve_visualization_config(
        cli_path=args.viz_config_yaml if str(args.viz_config_yaml).strip() else None,
        warnings=viz_warnings,
    )
    mesh_html_enabled = viz_enabled(viz_cfg, "mesh3d.html", True)
    compare_t8_enabled = viz_enabled(viz_cfg, "mesh3d.compare_t8", True)
    cfg_cutaway = viz_enabled(viz_cfg, "mesh3d.cutaway", True)
    cutaway_enabled = bool(args.enable_cutaway) if args.enable_cutaway is not None else bool(cfg_cutaway)

    model, _ = eval_holdout._load_model_from_state(Path(args.model_state))
    dataset = eval_holdout._load_dataset(Path(args.holdout_json))
    with Path(args.split_manifest).open("r", encoding="utf-8") as fp:
        split_manifest = json.load(fp)
    frame_path_map = eval_holdout._frame_paths_by_run(split_manifest)
    target_material_id = int(split_manifest.get("selected_target_material_id", 7))

    run = None
    if str(args.run_id).strip():
        for candidate in dataset.runs:
            if str(candidate.run_id) == str(args.run_id).strip():
                run = candidate
                break
    if run is None:
        run = dataset.runs[0]

    pred = rollout(run, model, simulation_options={"reinit_enabled": False})
    gt_np = [eval_holdout._to_numpy_zyx(frame) for frame in run.phi_t]
    pred_np = [eval_holdout._to_numpy_zyx(frame) for frame in pred]
    step_count = min(len(gt_np), len(pred_np))
    frames = _parse_frames(str(args.frames), step_count)

    frame_paths = frame_path_map.get(str(run.run_id), [])
    if frame_paths:
        valid_np = eval_holdout._load_valid_masks_from_paths(
            frame_paths=frame_paths[:step_count],
            valid_mask_array="ValidMask",
            fallback_shape=gt_np[0].shape,
        )
    else:
        valid_np = [np.ones_like(gt_np[idx], dtype=bool) for idx in range(step_count)]

    stride = max(1, int(args.voxel_stride))

    mesh_records: list[dict[str, Any]] = []
    gt_meshes: dict[int, MeshData] = {}
    pred_meshes: dict[int, MeshData] = {}
    gt_cut_meshes: dict[int, MeshData] = {}
    pred_cut_meshes: dict[int, MeshData] = {}

    for idx in frames:
        gt_phi = gt_np[idx]
        pd_phi = pred_np[idx]
        vm = np.asarray(valid_np[idx], dtype=bool)
        if stride > 1:
            gt_phi = gt_phi[::stride, ::stride, ::stride]
            pd_phi = pd_phi[::stride, ::stride, ::stride]
            vm = vm[::stride, ::stride, ::stride]
        gt_mask = _extract_hole_mask(gt_phi, vm, ref_center=None)
        gt_center = _component_center(gt_mask)
        pd_mask = _extract_hole_mask(pd_phi, vm, ref_center=gt_center)
        gt_mask_cut = _apply_quarter_cutaway(gt_mask, center=gt_center)
        pd_mask_cut = _apply_quarter_cutaway(pd_mask, center=gt_center)

        gt_mesh = _mask_to_surface_mesh(gt_mask)
        pd_mesh = _mask_to_surface_mesh(pd_mask)
        gt_cut_mesh = _mask_to_surface_mesh(gt_mask_cut)
        pd_cut_mesh = _mask_to_surface_mesh(pd_mask_cut)
        gt_meshes[idx] = gt_mesh
        pred_meshes[idx] = pd_mesh
        gt_cut_meshes[idx] = gt_cut_mesh
        pred_cut_meshes[idx] = pd_cut_mesh

        gt_ply = out_dir / f"mesh_gt_t{idx:04d}.ply"
        pd_ply = out_dir / f"mesh_pred_t{idx:04d}.ply"
        gt_cut_ply = out_dir / f"mesh_gt_cut_t{idx:04d}.ply"
        pd_cut_ply = out_dir / f"mesh_pred_cut_t{idx:04d}.ply"
        _write_ply(gt_ply, gt_mesh)
        _write_ply(pd_ply, pd_mesh)
        _write_ply(gt_cut_ply, gt_cut_mesh)
        _write_ply(pd_cut_ply, pd_cut_mesh)
        mesh_records.append(
            {
                "frame_index": int(idx),
                "gt_voxels": int(gt_mesh.voxel_count),
                "pred_voxels": int(pd_mesh.voxel_count),
                "gt_cut_voxels": int(gt_cut_mesh.voxel_count),
                "pred_cut_voxels": int(pd_cut_mesh.voxel_count),
                "gt_vertices": int(gt_mesh.vertices.shape[0]),
                "pred_vertices": int(pd_mesh.vertices.shape[0]),
                "gt_cut_vertices": int(gt_cut_mesh.vertices.shape[0]),
                "pred_cut_vertices": int(pd_cut_mesh.vertices.shape[0]),
                "gt_faces": int(gt_mesh.faces.shape[0]),
                "pred_faces": int(pd_mesh.faces.shape[0]),
                "gt_cut_faces": int(gt_cut_mesh.faces.shape[0]),
                "pred_cut_faces": int(pd_cut_mesh.faces.shape[0]),
                "gt_mesh_ply": str(gt_ply),
                "pred_mesh_ply": str(pd_ply),
                "gt_cut_mesh_ply": str(gt_cut_ply),
                "pred_cut_mesh_ply": str(pd_cut_ply),
            }
        )

    gt_full_color = "#14B8FF"
    pred_full_color = "#FF4D9A"
    gt_cut_color = "#00E5FF"
    pred_cut_color = "#FF8A3D"
    palette_cfg = viz_cfg.get("mesh3d", {}) if isinstance(viz_cfg.get("mesh3d"), dict) else {}
    if isinstance(palette_cfg.get("palette"), dict):
        palette_map = palette_cfg["palette"]
        gt_full_color = str(palette_map.get("gt_full", gt_full_color))
        pred_full_color = str(palette_map.get("pred_full", pred_full_color))
        gt_cut_color = str(palette_map.get("gt_cut", gt_cut_color))
        pred_cut_color = str(palette_map.get("pred_cut", pred_cut_color))

    cols = max(1, len(frames))
    html_path: Path | None = None
    t8_html_path: Path | None = None
    rows = 2 if cutaway_enabled else 1
    if mesh_html_enabled:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
            subplot_titles=(
                [f"t={i} full" for i in frames] + ([f"t={i} cutaway" for i in frames] if rows == 2 else [])
            ),
            horizontal_spacing=0.02,
            vertical_spacing=0.04,
        )
        for col, idx in enumerate(frames, start=1):
            gt_mesh = gt_meshes[idx]
            pd_mesh = pred_meshes[idx]
            if gt_mesh.faces.shape[0] > 0:
                fig.add_trace(
                    go.Mesh3d(
                        x=gt_mesh.vertices[:, 0],
                        y=gt_mesh.vertices[:, 1],
                        z=gt_mesh.vertices[:, 2],
                        i=gt_mesh.faces[:, 0],
                        j=gt_mesh.faces[:, 1],
                        k=gt_mesh.faces[:, 2],
                        color=gt_full_color,
                        opacity=0.34,
                        name="GT",
                        showlegend=bool(col == 1),
                        showscale=False,
                        flatshading=True,
                        lighting=dict(ambient=0.35, diffuse=0.65, specular=0.20, roughness=0.85, fresnel=0.08),
                        lightposition=dict(x=100, y=-120, z=180),
                    ),
                    row=1,
                    col=col,
                )
            if pd_mesh.faces.shape[0] > 0:
                fig.add_trace(
                    go.Mesh3d(
                        x=pd_mesh.vertices[:, 0],
                        y=pd_mesh.vertices[:, 1],
                        z=pd_mesh.vertices[:, 2],
                        i=pd_mesh.faces[:, 0],
                        j=pd_mesh.faces[:, 1],
                        k=pd_mesh.faces[:, 2],
                        color=pred_full_color,
                        opacity=0.34,
                        name="PRED",
                        showlegend=bool(col == 1),
                        showscale=False,
                        flatshading=True,
                        lighting=dict(ambient=0.35, diffuse=0.65, specular=0.20, roughness=0.85, fresnel=0.08),
                        lightposition=dict(x=-120, y=120, z=180),
                    ),
                    row=1,
                    col=col,
                )
            if rows == 2:
                gt_cut = gt_cut_meshes[idx]
                pd_cut = pred_cut_meshes[idx]
                if gt_cut.faces.shape[0] > 0:
                    fig.add_trace(
                        go.Mesh3d(
                            x=gt_cut.vertices[:, 0],
                            y=gt_cut.vertices[:, 1],
                            z=gt_cut.vertices[:, 2],
                            i=gt_cut.faces[:, 0],
                            j=gt_cut.faces[:, 1],
                            k=gt_cut.faces[:, 2],
                            color=gt_cut_color,
                            opacity=0.70,
                            name="GT cut",
                            showlegend=False,
                            showscale=False,
                            flatshading=True,
                            lighting=dict(ambient=0.28, diffuse=0.70, specular=0.25, roughness=0.80, fresnel=0.08),
                            lightposition=dict(x=100, y=-100, z=220),
                        ),
                        row=2,
                        col=col,
                    )
                if pd_cut.faces.shape[0] > 0:
                    fig.add_trace(
                        go.Mesh3d(
                            x=pd_cut.vertices[:, 0],
                            y=pd_cut.vertices[:, 1],
                            z=pd_cut.vertices[:, 2],
                            i=pd_cut.faces[:, 0],
                            j=pd_cut.faces[:, 1],
                            k=pd_cut.faces[:, 2],
                            color=pred_cut_color,
                            opacity=0.70,
                            name="PRED cut",
                            showlegend=False,
                            showscale=False,
                            flatshading=True,
                            lighting=dict(ambient=0.28, diffuse=0.70, specular=0.25, roughness=0.80, fresnel=0.08),
                            lightposition=dict(x=-100, y=100, z=220),
                        ),
                        row=2,
                        col=col,
                    )
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                scene_idx = (row - 1) * cols + col
                scene_name = "scene" if scene_idx == 1 else f"scene{scene_idx}"
                fig.update_layout(
                    **{
                        scene_name: dict(
                            xaxis=dict(visible=False, showbackground=True, backgroundcolor="rgb(245,245,250)"),
                            yaxis=dict(visible=False, showbackground=True, backgroundcolor="rgb(245,245,250)"),
                            zaxis=dict(visible=False, showbackground=True, backgroundcolor="rgb(245,245,250)"),
                            aspectmode="data",
                            camera=dict(eye=dict(x=1.55, y=1.55, z=1.15)),
                        )
                    }
                )

        fig.update_layout(
            title=(
                f"Hole Mesh Comparison (run={run.run_id}, target_material={target_material_id})"
                + (" - with cutaway" if rows == 2 else "")
            ),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        html_name = "hole_mesh_comparison_cutaway.html" if rows == 2 else "hole_mesh_comparison.html"
        html_path = out_dir / html_name
        fig.write_html(str(html_path), include_plotlyjs="cdn")

        if compare_t8_enabled:
            target_frame = 8 if 8 in gt_meshes else max(frames)
            gt_one = gt_meshes[target_frame]
            pd_one = pred_meshes[target_frame]
            gt_cut_one = gt_cut_meshes[target_frame]
            pd_cut_one = pred_cut_meshes[target_frame]
            fig_t8 = make_subplots(
                rows=2,
                cols=2,
                specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "scene"}, {"type": "scene"}]],
                subplot_titles=(f"GT t={target_frame} (full)", f"PRED t={target_frame} (full)", "GT cutaway", "PRED cutaway"),
                horizontal_spacing=0.03,
                vertical_spacing=0.06,
            )

            def _add_mesh(local_fig: Any, mesh: MeshData, *, row: int, col: int, color: str, opacity: float, name: str) -> None:
                if mesh.faces.shape[0] <= 0:
                    return
                local_fig.add_trace(
                    go.Mesh3d(
                        x=mesh.vertices[:, 0],
                        y=mesh.vertices[:, 1],
                        z=mesh.vertices[:, 2],
                        i=mesh.faces[:, 0],
                        j=mesh.faces[:, 1],
                        k=mesh.faces[:, 2],
                        color=color,
                        opacity=opacity,
                        name=name,
                        showlegend=False,
                        showscale=False,
                        flatshading=True,
                        lighting=dict(ambient=0.30, diffuse=0.68, specular=0.22, roughness=0.82, fresnel=0.08),
                    ),
                    row=row,
                    col=col,
                )

            _add_mesh(fig_t8, gt_one, row=1, col=1, color=gt_full_color, opacity=0.45, name="GT")
            _add_mesh(fig_t8, pd_one, row=1, col=2, color=pred_full_color, opacity=0.45, name="PRED")
            _add_mesh(fig_t8, gt_cut_one, row=2, col=1, color=gt_cut_color, opacity=0.72, name="GT cut")
            _add_mesh(fig_t8, pd_cut_one, row=2, col=2, color=pred_cut_color, opacity=0.72, name="PRED cut")

            for scene_idx in range(1, 5):
                scene_name = "scene" if scene_idx == 1 else f"scene{scene_idx}"
                fig_t8.update_layout(
                    **{
                        scene_name: dict(
                            xaxis=dict(visible=False, showbackground=True, backgroundcolor="rgb(245,245,250)"),
                            yaxis=dict(visible=False, showbackground=True, backgroundcolor="rgb(245,245,250)"),
                            zaxis=dict(visible=False, showbackground=True, backgroundcolor="rgb(245,245,250)"),
                            aspectmode="data",
                            camera=dict(eye=dict(x=1.55, y=1.55, z=1.15)),
                        )
                    }
                )
            fig_t8.update_layout(
                title=f"Hole Mesh Side-by-Side Compare (run={run.run_id}, t={target_frame})",
                template="plotly_white",
                margin=dict(l=0, r=0, t=70, b=0),
            )
            t8_html_path = out_dir / "hole_mesh_t8_side_by_side.html"
            fig_t8.write_html(str(t8_html_path), include_plotlyjs="cdn")
    else:
        viz_warnings.append("mesh html skipped: disabled by visualization config")

    manifest = {
        "run_id": str(run.run_id),
        "target_material_id": int(target_material_id),
        "frames": [int(v) for v in frames],
        "voxel_stride": int(stride),
        "cutaway_enabled": bool(cutaway_enabled),
        "html_enabled": bool(mesh_html_enabled),
        "compare_t8_enabled": bool(compare_t8_enabled),
        "html": None if html_path is None else str(html_path),
        "t8_compare_html": None if t8_html_path is None else str(t8_html_path),
        "palette": {
            "gt_full": gt_full_color,
            "pred_full": pred_full_color,
            "gt_cut": gt_cut_color,
            "pred_cut": pred_cut_color,
        },
        "records": mesh_records,
    }
    manifest_path = out_dir / "mesh_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    legacy_manifest_path = out_dir / "hole_mesh_manifest.json"
    with legacy_manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    viz_manifest_path = write_visualization_manifest(
        out_dir / "visualization_manifest.json",
        {
            "config": viz_cfg,
            "warnings": viz_warnings,
            "outputs": {
                "mesh_manifest_json": str(manifest_path),
                "html": None if html_path is None else str(html_path),
                "t8_compare_html": None if t8_html_path is None else str(t8_html_path),
            },
        },
    )

    print(f"mesh html: {html_path}")
    print(f"mesh t8 compare html: {t8_html_path}")
    print(f"mesh manifest: {manifest_path}")
    print(f"visualization manifest: {viz_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
