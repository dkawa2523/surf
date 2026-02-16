#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wafer_surrogate.config.loader import load_config


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def _csv(value: Any, default: str) -> str:
    out = _as_list(value)
    return ",".join(out) if out else default


def _str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _num(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _bool01(value: Any, default: str = "0") -> str:
    if value is None:
        return default
    return "1" if bool(value) else "0"


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    print(f"[yaml-runner] exec: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))


def _read_run_id(config_path: Path) -> str:
    payload = load_config(config_path)
    run = _mapping(payload.get("run"))
    run_id = _str(run.get("run_id"), "")
    return run_id if run_id else "dataset_3d_test2_pilot_quick"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run-dataset-3d-test2-from-yaml",
        description="Run dataset_3d_test2 pilot flow from a single YAML config.",
    )
    parser.add_argument(
        "--config",
        default="configs/dataset_3d_test2_train_eval.yaml",
        help="YAML config path for prepare/train/eval settings.",
    )
    parser.add_argument(
        "--skip-mesh",
        action="store_true",
        help="Skip mesh 3D postprocess even when enabled in YAML.",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists() or not cfg_path.is_file():
        raise ValueError(f"config not found: {cfg_path}")
    cfg = load_config(cfg_path)

    paths = _mapping(cfg.get("paths"))
    split = _mapping(cfg.get("split"))
    prepare = _mapping(cfg.get("prepare"))
    evaluation = _mapping(cfg.get("evaluation"))
    runtime = _mapping(cfg.get("runtime"))
    postprocess = _mapping(cfg.get("postprocess"))
    mesh_cfg = _mapping(postprocess.get("mesh3d"))

    train_config_path = Path(_str(paths.get("train_config"), "configs/dataset_3d_test2_pilot_train.yaml"))
    if not train_config_path.is_absolute():
        train_config_path = (ROOT / train_config_path).resolve()
    if not train_config_path.exists() or not train_config_path.is_file():
        raise ValueError(f"train_config not found: {train_config_path}")

    env = dict(os.environ)
    env["DATA_DIR"] = _str(paths.get("data_dir"), "ataset_3d_test2")
    env["OUT_ROOT"] = _str(paths.get("out_root"), "runs/dataset_3d_test2_pilot")
    env["BASE_CONFIG"] = str(train_config_path)
    env["VIZ_CONFIG"] = _str(paths.get("viz_config"), "configs/visualization.default.yaml")
    env["BASELINE_EVAL_JSON"] = _str(
        paths.get("baseline_eval_json"),
        "runs/dataset_3d_test2_pilot_h0006/eval/quick/holdout_eval.json",
    )

    default_train_runs = ",".join([f"run_{idx:04d}" for idx in range(12)])
    env["TRAIN_RUNS"] = _csv(split.get("train_runs"), default_train_runs)
    env["PRIMARY_HOLDOUT_RUN"] = _str(split.get("primary_holdout_run"), "run_0006")
    env["SECONDARY_HOLDOUT_RUN"] = _str(split.get("secondary_holdout_run"), "run_0012")

    env["VTI_PATTERN"] = _str(prepare.get("vti_pattern"), "vox_t*.vti")
    env["BAND_WIDTH"] = _num(prepare.get("band_width"), "2.0")
    env["MIN_GRAD_NORM"] = _num(prepare.get("min_grad_norm"), "1e-8")
    env["VALID_MASK_ARRAY"] = _str(prepare.get("valid_mask_array"), "ValidMask")
    env["TARGET_MATERIAL_ID"] = _str(prepare.get("target_material_id"), "")
    env["TARGET_SELECTION_MODE"] = _str(prepare.get("target_selection_mode"), "")
    env["TARGET_MAX_RATIO"] = _num(prepare.get("target_max_ratio"), "")
    env["TARGET_MAX_BOUNDARY_RATIO"] = _num(prepare.get("target_max_boundary_ratio"), "")
    env["TARGET_MIN_TEMPORAL_DELTA"] = _num(prepare.get("target_min_temporal_delta"), "")
    env["DOMAIN_BOUNDARY_MARGIN_VOX"] = _num(prepare.get("domain_boundary_margin_vox"), "1")
    env["PHI_BOUNDARY_CLIP_VOX"] = _num(prepare.get("phi_boundary_clip_vox"), "3")
    env["INCLUDE_TERMINAL_STEP_TARGET"] = _bool01(prepare.get("include_terminal_step_target"), "0")
    env["MASK_ARRAY"] = _str(prepare.get("mask_array"), "")
    env["MASK_MATERIAL_ID"] = _num(prepare.get("mask_material_id"), "")

    env["ANALYSIS_XY_MARGIN_VOX"] = _num(evaluation.get("analysis_xy_margin_vox"), "2")
    env["ANALYSIS_Z_MARGIN_VOX"] = _num(evaluation.get("analysis_z_margin_vox"), "1")

    env["RUN_REAL_ME"] = _bool01(runtime.get("run_real_me"), "0")

    _run(["bash", "scripts/run_dataset_3d_test2_pilot.sh"], env=env)

    mesh_enabled = bool(mesh_cfg.get("enabled", False)) and (not args.skip_mesh)
    if mesh_enabled:
        run_id = _read_run_id(train_config_path)
        model_state = ROOT / "runs" / run_id / "train" / "outputs" / "model_state.json"
        split_manifest = Path(env["OUT_ROOT"]) / "prepared" / "split_manifest.json"
        holdout_json = Path(env["OUT_ROOT"]) / "prepared" / "holdout_dataset.json"
        mesh_out_dir = Path(env["OUT_ROOT"]) / "eval" / "quick_primary" / "mesh3d"
        if model_state.exists() and split_manifest.exists() and holdout_json.exists():
            cmd = [
                sys.executable,
                "scripts/visualize_hole_mesh_3d.py",
                "--model-state",
                str(model_state),
                "--holdout-json",
                str(holdout_json),
                "--split-manifest",
                str(split_manifest),
                "--out-dir",
                str(mesh_out_dir),
                "--frames",
                _str(mesh_cfg.get("frames"), "0,4,8"),
                "--voxel-stride",
                _num(mesh_cfg.get("voxel_stride"), "1"),
                "--viz-config-yaml",
                env["VIZ_CONFIG"],
            ]
            if bool(mesh_cfg.get("enable_cutaway", True)):
                cmd.append("--enable-cutaway")
            else:
                cmd.append("--disable-cutaway")
            _run(cmd, env=env)
        else:
            print(
                "[yaml-runner] mesh postprocess skipped: required artifacts are missing",
                flush=True,
            )

    report_path = Path(env["OUT_ROOT"]) / "eval" / "report_index.json"
    print(f"[yaml-runner] done. report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
