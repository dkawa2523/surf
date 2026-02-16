from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wafer_surrogate.data.synthetic import generate_synthetic_sdf_dataset

from .common import now_utc, safe_run_id, write_smoke_metric_logs


def _latest_matching_file(directory: Path, pattern: str) -> Path | None:
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, Mapping):
        raise ValueError(f"json payload must be a mapping: {path}")
    return payload


def _load_phi_series_from_run_dir(run_dir: Path) -> tuple[str, float, list[Any]]:
    if not run_dir.exists() or not run_dir.is_dir():
        raise ValueError(f"run-dir does not exist: {run_dir}")
    infer_path = _latest_matching_file(run_dir, "infer_output_*.json")
    if infer_path is not None:
        payload = _load_json_mapping(infer_path)
        phi_t = payload.get("phi_t")
        if not isinstance(phi_t, Sequence) or isinstance(phi_t, (str, bytes, bytearray)) or not phi_t:
            raise ValueError(f"phi_t is missing or empty in {infer_path}")
        return str(payload.get("run_id", run_dir.name)), float(payload.get("dt", 1.0)), list(phi_t)
    synthetic_path = _latest_matching_file(run_dir, "synthetic_sdf_*.json")
    if synthetic_path is not None:
        payload = _load_json_mapping(synthetic_path)
        runs = payload.get("runs")
        if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes, bytearray)) or not runs:
            raise ValueError(f"runs is missing or empty in {synthetic_path}")
        first_run = runs[0]
        if not isinstance(first_run, Mapping):
            raise ValueError(f"runs[0] must be a mapping in {synthetic_path}")
        phi_t = first_run.get("phi_t")
        if not isinstance(phi_t, Sequence) or isinstance(phi_t, (str, bytes, bytearray)) or not phi_t:
            raise ValueError(f"phi_t is missing or empty in {synthetic_path}")
        return str(first_run.get("run_id", run_dir.name)), float(first_run.get("dt", 1.0)), list(phi_t)
    raise ValueError(f"run-dir does not contain infer_output_*.json or synthetic_sdf_*.json: {run_dir}")


def _load_phi_series_from_npz(phi_npz: Path) -> tuple[str, float, list[Any]]:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("--phi-npz requires numpy") from exc
    with np.load(phi_npz, allow_pickle=False) as npz:
        if "phi_t" in npz.files:
            phi_array = npz["phi_t"]
        elif len(npz.files) == 1:
            phi_array = npz[npz.files[0]]
        else:
            raise ValueError("--phi-npz must include key 'phi_t' when multiple arrays are stored")
        if int(phi_array.ndim) not in (3, 4):
            raise ValueError("phi_t in npz must be 3D or 4D with a leading time axis")
        phi_t = [phi_array[step] for step in range(int(phi_array.shape[0]))]
        dt = 1.0
        if "dt" in npz.files:
            dt = float(np.asarray(npz["dt"]).reshape(-1)[0])
        run_id = phi_npz.stem
        if "run_id" in npz.files:
            run_id_raw: Any = npz["run_id"]
            if hasattr(run_id_raw, "item"):
                run_id_raw = run_id_raw.item()
            if isinstance(run_id_raw, bytes):
                run_id_raw = run_id_raw.decode("utf-8", errors="ignore")
            run_id = str(run_id_raw)
    return run_id, dt, phi_t


def _infer_run_id_from_vti_dir(vti_dir: Path) -> str:
    if vti_dir.name == "phi" and vti_dir.parent.name == "vti" and vti_dir.parent.parent.name == "viz":
        candidate = vti_dir.parent.parent.parent.name
        if candidate:
            return candidate
    return vti_dir.name


def _output_root_from_vti_dir(vti_dir: Path, run_id: str) -> Path:
    if vti_dir.name == "phi" and vti_dir.parent.name == "vti" and vti_dir.parent.parent.name == "viz":
        return vti_dir.parent.parent.parent
    return Path("runs") / safe_run_id(run_id)


def _load_phi_series_from_vti_dir(vti_dir: Path) -> tuple[str, float, list[Any]]:
    from wafer_surrogate.viz.render import load_vti_series

    run_id = _infer_run_id_from_vti_dir(vti_dir)
    phi_t = load_vti_series(vti_dir, pattern="phi_t*.vti", array_name="phi")
    return run_id, 1.0, phi_t


def cmd_viz_export_vti(args: argparse.Namespace) -> int:
    selected_inputs = int(bool(args.smoke)) + int(args.run_dir is not None) + int(args.phi_npz is not None)
    if selected_inputs != 1:
        raise ValueError("choose exactly one input source: --smoke or --run-dir or --phi-npz")
    if args.smoke:
        dataset = generate_synthetic_sdf_dataset(num_runs=1, num_steps=4, grid_size=24, dt=0.1)
        source_run = dataset.runs[0]
        run_id = f"{source_run.run_id}_{now_utc()}"
        dt = float(source_run.dt)
        phi_t = source_run.phi_t
        output_root = Path("runs") / safe_run_id(run_id)
    elif args.run_dir is not None:
        source_run_dir = Path(args.run_dir)
        run_id, dt, phi_t = _load_phi_series_from_run_dir(source_run_dir)
        safe = safe_run_id(run_id)
        output_root = source_run_dir if source_run_dir.name == safe else (source_run_dir / safe)
    else:
        run_id, dt, phi_t = _load_phi_series_from_npz(Path(args.phi_npz))
        output_root = Path("runs") / safe_run_id(run_id)
    if not isinstance(phi_t, Sequence) or isinstance(phi_t, (str, bytes, bytearray)) or not phi_t:
        raise ValueError("phi_t must contain at least one frame")
    from wafer_surrogate.viz.vti import export_vti_series

    times = [float(step) * float(dt) for step in range(len(phi_t))]
    field_dir = output_root / "viz" / "vti" / "phi"
    result = export_vti_series(
        out_dir=field_dir,
        frames=phi_t,
        array_name="phi",
        file_prefix="phi",
        times=times,
    )
    print(f"vti directory written: {result['out_dir']}")
    print(f"pvd written: {result['pvd_path']}")
    print(f"exported frames: {len(result['vti_paths'])}")
    return 0


def cmd_viz_render_slices(args: argparse.Namespace) -> int:
    selected_inputs = (
        int(bool(args.smoke))
        + int(args.run_dir is not None)
        + int(args.phi_npz is not None)
        + int(args.vti_dir is not None)
    )
    if selected_inputs != 1:
        raise ValueError("choose exactly one input source: --smoke or --run-dir or --phi-npz or --vti-dir")
    if args.smoke:
        dataset = generate_synthetic_sdf_dataset(num_runs=1, num_steps=4, grid_size=24, dt=0.1)
        source_run = dataset.runs[0]
        run_id = f"{source_run.run_id}_{now_utc()}"
        dt = float(source_run.dt)
        phi_t = source_run.phi_t
        output_root = Path("runs") / safe_run_id(run_id)
    elif args.run_dir is not None:
        source_run_dir = Path(args.run_dir)
        run_id, dt, phi_t = _load_phi_series_from_run_dir(source_run_dir)
        safe = safe_run_id(run_id)
        output_root = source_run_dir if source_run_dir.name == safe else (source_run_dir / safe)
    elif args.vti_dir is not None:
        source_vti_dir = Path(args.vti_dir)
        run_id, dt, phi_t = _load_phi_series_from_vti_dir(source_vti_dir)
        output_root = _output_root_from_vti_dir(source_vti_dir, run_id)
    else:
        run_id, dt, phi_t = _load_phi_series_from_npz(Path(args.phi_npz))
        output_root = Path("runs") / safe_run_id(run_id)
    if not isinstance(phi_t, Sequence) or isinstance(phi_t, (str, bytes, bytearray)) or not phi_t:
        raise ValueError("phi_t must contain at least one frame")
    from wafer_surrogate.viz.render import render_slice_quicklook_series

    slices_dir = output_root / "viz" / "png" / "slices"
    result = render_slice_quicklook_series(
        frames=phi_t,
        out_dir=slices_dir,
        x_index=args.x_index,
        y_index=args.y_index,
        z_index=args.z_index,
        contour_level=float(args.contour_level),
        file_prefix="slices",
    )
    print(f"slice directory written: {result['out_dir']}")
    print(f"slice shape: {result['shape']}")
    print(f"slice indices: {result['slice_indices']}")
    if result["matplotlib_available"]:
        print(f"rendered png frames: {len(result['png_paths'])}")
        return 0
    if args.vti_dir is None:
        from wafer_surrogate.viz.vti import export_vti_series

        fallback_vti = export_vti_series(
            out_dir=output_root / "viz" / "vti" / "phi",
            frames=phi_t,
            array_name="phi",
            file_prefix="phi",
            times=[float(step) * float(dt) for step in range(len(phi_t))],
        )
        print(f"matplotlib unavailable; exported VTI fallback: {fallback_vti['out_dir']}")
    else:
        print("matplotlib unavailable; keeping provided VTI input as fallback artifact.")
    print(f"fallback slice files: {len(result['fallback_paths'])}")
    return 0


def cmd_viz_compare_sections(args: argparse.Namespace) -> int:
    if not bool(args.smoke):
        raise ValueError("P0 compare-sections currently supports --smoke only")
    dataset = generate_synthetic_sdf_dataset(num_runs=1, num_steps=4, grid_size=24, dt=0.1)
    run = dataset.runs[0]
    gt_field = run.phi_t[-1]
    pred_field = [[float(value) - 0.3 for value in row] for row in gt_field]
    run_id = f"{run.run_id}_{now_utc()}"
    out_dir = Path("runs") / safe_run_id(run_id) / "viz" / "png" / "compare"
    from wafer_surrogate.viz.compare_sections import render_compare_section

    result = render_compare_section(
        pred_field=pred_field,
        gt_field=gt_field,
        out_dir=out_dir,
        split=str(args.split),
        sample_index=0,
        y_index=args.y_index,
        contour_level=float(args.contour_level),
        frame_index=len(run.phi_t) - 1,
    )
    print(f"compare directory written: {result['out_dir']}")
    print(f"compare y-index: {result['y_index']}")
    if result["matplotlib_available"]:
        print(f"compare png written: {result['png_path']}")
    else:
        print(f"matplotlib unavailable; contour fallback written: {result['fallback_path']}")
    return 0


def cmd_viz_plot_metrics(args: argparse.Namespace) -> int:
    selected_inputs = int(bool(args.smoke)) + int(args.run_dir is not None)
    if selected_inputs != 1:
        raise ValueError("choose exactly one input source: --smoke or --run-dir")
    if bool(args.smoke):
        run_id = f"plot_metrics_smoke_{now_utc()}"
        source_run_dir = Path("runs") / safe_run_id(run_id)
        write_smoke_metric_logs(source_run_dir)
        run_id_hint: str | None = run_id
    else:
        source_run_dir = Path(args.run_dir)
        run_id_hint = None
    from wafer_surrogate.viz.plots import plot_metrics_for_run_dir

    result = plot_metrics_for_run_dir(
        run_dir=source_run_dir,
        run_id_hint=run_id_hint,
    )
    print(f"plot-metrics directory written: {result['plots_dir']}")
    print(f"plot-metrics source files: {result['num_source_files']}")
    print(f"plot-metrics metrics: {result['num_metrics']}")
    if result["matplotlib_available"]:
        print(f"plot-metrics curves written: {result['curves_png_path']}")
        print(f"plot-metrics histogram written: {result['hist_png_path']}")
    else:
        print(f"matplotlib unavailable; csv fallback written: {result['points_csv_path']}")
    return 0


def cmd_viz_leaderboard(args: argparse.Namespace) -> int:
    from wafer_surrogate.viz.leaderboard import render_leaderboard_for_run

    run_dir = Path(args.run_dir)
    result = render_leaderboard_for_run(run_dir)
    print(f"leaderboard viz directory written: {result['viz_dir']}")
    print(f"leaderboard records: {result['num_records']}")
    if bool(result["matplotlib_available"]):
        print(f"leaderboard scores plot: {result['score_png_path']}")
        if result.get("scatter_png_path"):
            print(f"leaderboard mae-rmse plot: {result['scatter_png_path']}")
    else:
        print(f"matplotlib unavailable; csv fallback written: {result['long_csv_path']}")
    return 0
