from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

from wafer_surrogate.data.synthetic import SyntheticSDFDataset, SyntheticSDFRun, generate_synthetic_sdf_dataset
from wafer_surrogate.features import list_feature_extractors
from wafer_surrogate.inference import assess_ood
from wafer_surrogate.metrics import format_metrics_json
from wafer_surrogate.models import make_model
from wafer_surrogate.optimization import run_optimization_engine
from wafer_surrogate.pipeline import run_pipeline as run_stage_pipeline

from .common import (
    batch_conditions,
    build_dataset,
    build_model,
    build_prior,
    eval_metrics,
    fit_model,
    flatten_condition_row,
    frame_mean,
    inference_conditions,
    load_config,
    make_samples_with_pipeline,
    maps_equal,
    now_utc,
    ood_threshold,
    output_dir,
    prior_preview,
    rollout,
    search_ranges,
    split_eval_samples,
    to_float_map,
    write_csv_timestamped,
    write_eval_compare_sections,
    write_json_timestamped,
    write_plot,
)


def _artifact_index(manifest: Any, stage_name: str) -> dict[str, Path]:
    artifacts = getattr(manifest, "stage_artifacts", {}).get(stage_name, [])
    out: dict[str, Path] = {}
    for ref in artifacts:
        name = getattr(ref, "name", "")
        path = getattr(ref, "path", "")
        if not name or not path:
            continue
        out[str(name)] = Path(str(path))
    return out


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"json payload must be mapping: {path}")
    return payload


def _pipeline_override(base: Mapping[str, Any] | None, patch: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base) if isinstance(base, Mapping) else {}
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[str(key)] = _pipeline_override(out.get(str(key), {}), value)
        else:
            out[str(key)] = value
    return out


def _eval_data_override(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(getattr(args, "smoke", False)):
        return {}
    return {
        "data": {
            "source": "synthetic",
            "num_runs": 4,
            "num_steps": 4,
            "grid_size": 20,
            "dt": 0.1,
        }
    }


def _to_plot_frame_2d(frame: Any) -> list[list[float]]:
    if not isinstance(frame, list) or not frame:
        raise ValueError("plot frame must be a non-empty sequence")
    first = frame[0]
    if not isinstance(first, list) or not first:
        raise ValueError("plot frame must be a non-empty 2D/3D sequence")
    if isinstance(first[0], list):
        z_index = max(0, len(frame) // 2)
        slice_2d = frame[z_index]
        if not isinstance(slice_2d, list) or not slice_2d:
            raise ValueError("3D plot frame has invalid middle slice")
        return [[float(cell) for cell in row] for row in slice_2d]
    return [[float(cell) for cell in row] for row in frame]


def _shift_frame(frame: list[list[float]], delta: float) -> list[list[float]]:
    return [[float(cell) + float(delta) for cell in row] for row in frame]


def _resolve_rollout_preview(
    *,
    inference_artifacts: Mapping[str, Path],
    infer_payload: Mapping[str, Any],
    run_dir: Path,
) -> tuple[list[list[float]], list[list[float]]] | None:
    candidates: list[Path] = []
    artifact_path = inference_artifacts.get("inference_single_rollout_preview")
    if artifact_path is not None:
        candidates.append(Path(artifact_path))
    payload_path = infer_payload.get("rollout_preview_path")
    if isinstance(payload_path, str) and payload_path.strip():
        p = Path(payload_path)
        if not p.is_absolute():
            p = (run_dir / p).resolve()
        candidates.append(p)
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        loaded = _read_json(path)
        initial = loaded.get("initial_frame")
        final = loaded.get("final_frame")
        if isinstance(initial, list) and isinstance(final, list):
            return _to_plot_frame_2d(initial), _to_plot_frame_2d(final)
    return None


def _select_dataset_run(dataset: SyntheticSDFDataset, run_id: str | None) -> SyntheticSDFRun:
    if run_id:
        for run in dataset.runs:
            if str(run.run_id) == str(run_id):
                return run
    if not dataset.runs:
        raise ValueError("dataset is empty")
    return dataset.runs[0]


def _run_pipeline_compat(
    *,
    args: argparse.Namespace,
    stages: list[str],
    override: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any], Path, Any, Path]:
    config_path = Path(args.config)
    config = load_config(config_path)
    out_dir = output_dir(config)
    merged_override = _pipeline_override({}, override or {})
    manifest = run_stage_pipeline(config_path=config_path, selected_stages=stages, config_override=merged_override)
    run_dir = out_dir / manifest.run_id
    return config_path, config, out_dir, manifest, run_dir


def cmd_train_distill(args: argparse.Namespace) -> int:
    print(
        "warning: 'train-distill' is deprecated and will be removed after 2026-06-30; "
        "use `pipeline run --stages cleaning,featurization,preprocessing,train` with `train.mode=sparse_distill`."
    )
    config_path, config, out_dir, manifest, run_dir = _run_pipeline_compat(
        args=args,
        stages=["cleaning", "featurization", "preprocessing", "train"],
        override={
            "featurization": {
                "target_mode": "vn_narrow_band",
            },
            "train": {
                "mode": "sparse_distill",
            },
        },
    )
    train_metrics = dict(manifest.stage_metrics.get("train", {}))
    train_artifacts = _artifact_index(manifest, "train")
    distill_metrics_payload: dict[str, Any] = {}
    distill_path = train_artifacts.get("distill_metrics")
    if distill_path is not None and distill_path.exists():
        distill_metrics_payload = _read_json(distill_path)
    distill_summary = {
        "teacher_metrics": {
            "mae": float(distill_metrics_payload.get("teacher_mae", train_metrics.get("teacher_mae", 0.0)))
        },
        "student_metrics": {
            "mae": float(
                distill_metrics_payload.get(
                    "student_mae",
                    train_metrics.get("student_mae", train_metrics.get("mae", 0.0)),
                )
            )
        },
        "distill_gap": float(distill_metrics_payload.get("distill_gap", train_metrics.get("distill_gap", 0.0))),
        "run_id": manifest.run_id,
        "manifest_path": str(run_dir / "manifest.json"),
        "train_mode": "sparse_distill",
        "artifacts": {name: str(path) for name, path in train_artifacts.items()},
        "distill_metrics": distill_metrics_payload,
    }
    summary_path = write_json_timestamped(
        out_dir,
        "train_distill_summary",
        {
            "artifact_type": "train_distill_summary",
            "created_at_utc": now_utc(),
            "config_path": str(config_path),
            "data_source": str(config.get("data", {}).get("source", "synthetic")) if isinstance(config.get("data"), Mapping) else "synthetic",
            "run_id": manifest.run_id,
            "manifest_path": str(run_dir / "manifest.json"),
            "distillation": distill_summary,
        },
    )
    print(f"train-distill summary written: {summary_path}")
    print(
        "train-distill metrics: "
        f"teacher_mae={distill_summary['teacher_metrics']['mae']:.6f}, "
        f"student_mae={distill_summary['student_metrics']['mae']:.6f}"
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    override: dict[str, Any] = {}
    if bool(getattr(args, "smoke", False)):
        override = {
            "data": {
                "source": "synthetic",
                "num_runs": 1,
                "num_steps": 4,
                "grid_size": 20,
                "dt": 0.1,
            }
        }
    config_path, config, out_dir, manifest, run_dir = _run_pipeline_compat(
        args=args,
        stages=["cleaning", "featurization", "preprocessing", "train"],
        override=override,
    )
    train_metrics = dict(manifest.stage_metrics.get("train", {}))
    train_artifacts = _artifact_index(manifest, "train")
    model_name = "unknown"
    if train_artifacts.get("train_metrics") and train_artifacts["train_metrics"].exists():
        train_payload = _read_json(train_artifacts["train_metrics"])
        model_name = str(train_payload.get("best_model", "unknown"))
    prior_name, prior_kwargs, prior = build_prior(config)
    summary_path = write_json_timestamped(
        out_dir,
        "train_summary",
        {
            "artifact_type": "train_summary",
            "created_at_utc": now_utc(),
            "config_path": str(config_path),
            "data_source": str(config.get("data", {}).get("source", "synthetic")) if isinstance(config.get("data"), Mapping) else "synthetic",
            "run_id": manifest.run_id,
            "manifest_path": str(run_dir / "manifest.json"),
            "model_name": model_name,
            "prior": {"name": prior_name, "kwargs": prior_kwargs, **prior_preview(prior)},
            "metrics": train_metrics,
        },
    )
    print(f"train summary written: {summary_path}")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    config_path, config, out_dir, manifest, run_dir = _run_pipeline_compat(
        args=args,
        stages=["cleaning", "featurization", "preprocessing", "train", "inference"],
        override={"inference": {"mode": "single"}},
    )
    inference_artifacts = _artifact_index(manifest, "inference")
    if "inference_single" not in inference_artifacts:
        raise ValueError("infer pipeline output missing inference_single artifact")
    infer_payload = _read_json(inference_artifacts["inference_single"])
    prediction = infer_payload.get("prediction", {}) if isinstance(infer_payload.get("prediction"), Mapping) else {}
    infer_cond = (
        {str(k): float(v) for k, v in infer_payload.get("conditions", {}).items()}
        if isinstance(infer_payload.get("conditions"), Mapping)
        else {}
    )
    ood_report = infer_payload.get("ood", {})
    infer_metrics: dict[str, float] | None = None
    if "predicted_target" in prediction:
        infer_metrics = {"predicted_target": float(prediction["predicted_target"])}
    elif "final_phi_mean" in prediction:
        infer_metrics = {"final_phi_mean": float(prediction["final_phi_mean"])}
    plot_path = None
    plot_error: str | None = None
    if bool(getattr(args, "plot", False)):
        try:
            frames = _resolve_rollout_preview(
                inference_artifacts=inference_artifacts,
                infer_payload=infer_payload,
                run_dir=run_dir,
            )
            if frames is not None:
                base_frame, final_frame = frames
            else:
                dataset = build_dataset(config)
                if not dataset.runs or not dataset.runs[0].phi_t:
                    raise ValueError("dataset is empty")
                base_frame = _to_plot_frame_2d(dataset.runs[0].phi_t[0])
                final_frame = base_frame
                if "final_phi_mean" in prediction:
                    delta = float(prediction["final_phi_mean"]) - float(frame_mean(base_frame))
                    final_frame = _shift_frame(base_frame, delta)
                elif "predicted_target" in prediction:
                    num_steps = max(1, int(prediction.get("num_steps", 1)))
                    dt = float(dataset.runs[0].dt)
                    final_frame = _shift_frame(base_frame, -(float(prediction["predicted_target"]) * dt * float(num_steps)))
            plot_path = write_plot(out_dir, [base_frame, final_frame])
        except Exception as exc:
            plot_error = str(exc)
    summary_path = write_json_timestamped(
        out_dir,
        "infer_output",
        {
            "artifact_type": "infer_output",
            "created_at_utc": now_utc(),
            "config_path": str(config_path),
            "data_source": str(config.get("data", {}).get("source", "synthetic")) if isinstance(config.get("data"), Mapping) else "synthetic",
            "run_id": manifest.run_id,
            "manifest_path": str(run_dir / "manifest.json"),
            "conditions": infer_cond,
            "prediction": prediction,
            "ood": ood_report,
            "reference_metrics": infer_metrics,
            "plot_path": None if plot_path is None else str(plot_path),
            "plot_error": plot_error,
        },
    )
    print(f"infer output written: {summary_path}")
    ood_status = str(ood_report.get("status", "unknown")) if isinstance(ood_report, Mapping) else "unknown"
    ood_distance = None
    ood_threshold = None
    if isinstance(ood_report, Mapping):
        condition = ood_report.get("condition")
        if isinstance(condition, Mapping):
            ood_distance = condition.get("distance")
            ood_threshold = condition.get("threshold")
    print(f"infer ood status: {ood_status} (distance={ood_distance}, threshold={ood_threshold})")
    if infer_metrics is not None and isinstance(infer_metrics, Mapping):
        print(f"infer metrics: {format_metrics_json(infer_metrics)}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    override: dict[str, Any] = {"inference": {"mode": "single"}}
    smoke_override = _eval_data_override(args)
    if smoke_override:
        override = _pipeline_override(override, smoke_override)
    config_path, config, out_dir, manifest, run_dir = _run_pipeline_compat(
        args=args,
        stages=["cleaning", "featurization", "preprocessing", "train", "inference"],
        override=override,
    )
    metrics = dict(manifest.stage_metrics.get("train", {}))
    inference_artifacts = _artifact_index(manifest, "inference")
    infer_payload = (
        _read_json(inference_artifacts["inference_single"])
        if inference_artifacts.get("inference_single") and inference_artifacts["inference_single"].exists()
        else {}
    )
    compare_summary: dict[str, Any]
    if bool(getattr(args, "compare_sections", True)):
        try:
            compare_cfg = _pipeline_override(config, smoke_override) if smoke_override else dict(config)
            dataset = build_dataset(compare_cfg)
            run_id_hint = str(infer_payload.get("template_run_id", "")) if isinstance(infer_payload, Mapping) else ""
            gt_run = _select_dataset_run(dataset, run_id_hint)
            preview = _resolve_rollout_preview(
                inference_artifacts=inference_artifacts,
                infer_payload=infer_payload if isinstance(infer_payload, Mapping) else {},
                run_dir=run_dir,
            )
            if preview is not None:
                initial_frame, final_frame = preview
                split_samples = {"val": [(gt_run, [initial_frame, final_frame])]}
                compare_source = "pipeline_inference_preview"
            else:
                _, _, model = build_model(compare_cfg)
                fit_model(model, dataset)
                predicted_runs = [rollout(run, model) for run in dataset.runs]
                split_samples = split_eval_samples(dataset, predicted_runs)
                compare_source = "compat_rollout_fallback"
            compare_summary = write_eval_compare_sections(
                out_dir=out_dir,
                split_samples=split_samples,
                max_samples=max(0, int(getattr(args, "compare_max_samples", 3))),
                y_index=getattr(args, "y_index", None),
                contour_level=0.0,
            )
            compare_summary["source"] = compare_source
            compare_summary["run_id"] = manifest.run_id
        except Exception as exc:
            compare_summary = {
                "enabled": True,
                "generated": 0,
                "records": [],
                "error": str(exc),
                "note": "compare-sections failed",
            }
    else:
        compare_summary = {
            "enabled": False,
            "generated": 0,
            "records": [],
            "note": "compare-sections disabled by CLI option",
        }
    prior_name, prior_kwargs, prior = build_prior(config)
    model_name = "unknown"
    train_artifacts = _artifact_index(manifest, "train")
    if train_artifacts.get("train_metrics") and train_artifacts["train_metrics"].exists():
        train_payload = _read_json(train_artifacts["train_metrics"])
        model_name = str(train_payload.get("best_model", "unknown"))
    summary_path = write_json_timestamped(
        out_dir,
        "eval_summary",
        {
            "artifact_type": "eval_summary",
            "created_at_utc": now_utc(),
            "config_path": str(config_path),
            "data_source": str(config.get("data", {}).get("source", "synthetic")) if isinstance(config.get("data"), Mapping) else "synthetic",
            "model_name": model_name,
            "prior": {"name": prior_name, "kwargs": prior_kwargs, **prior_preview(prior)},
            "run_id": manifest.run_id,
            "manifest_path": str(run_dir / "manifest.json"),
            "metrics": metrics,
            "compare_sections": compare_summary,
        },
    )
    print(f"eval summary written: {summary_path}")
    print(f"eval metrics: {format_metrics_json(metrics)}")
    if compare_summary["enabled"]:
        print(f"eval compare-sections generated: {compare_summary['generated']}")
    if bool(getattr(args, "plot_metrics", True)):
        from wafer_surrogate.viz.plots import plot_metrics_for_run_dir

        run_id_hint = manifest.run_id
        plot_summary = plot_metrics_for_run_dir(run_dir=run_dir, run_id_hint=run_id_hint)
        print(f"eval plot-metrics directory: {plot_summary['plots_dir']}")
        if not bool(plot_summary["matplotlib_available"]):
            print(
                "matplotlib unavailable; plot-metrics wrote CSV fallback: "
                f"{plot_summary['points_csv_path']}"
            )
    return 0


def cmd_infer_batch(args: argparse.Namespace) -> int:
    config_path, config, out_dir, manifest, run_dir = _run_pipeline_compat(
        args=args,
        stages=["cleaning", "featurization", "preprocessing", "train", "inference"],
        override={"inference": {"mode": "batch", "batch_size": max(1, int(args.batch_size))}},
    )
    inference_artifacts = _artifact_index(manifest, "inference")
    if "inference_batch_summary" not in inference_artifacts:
        raise ValueError("infer-batch pipeline output missing inference_batch_summary artifact")
    batch_payload = _read_json(inference_artifacts["inference_batch_summary"])
    rows = [dict(row) for row in batch_payload.get("rows", [])] if isinstance(batch_payload.get("rows"), list) else []
    csv_path = inference_artifacts.get("inference_batch_csv")

    model_name = "unknown"
    train_artifacts = _artifact_index(manifest, "train")
    if train_artifacts.get("train_metrics") and train_artifacts["train_metrics"].exists():
        model_name = str(_read_json(train_artifacts["train_metrics"]).get("best_model", "unknown"))

    objectives = [float(row.get("objective", 0.0)) for row in rows]
    summary_path = write_json_timestamped(
        out_dir=out_dir,
        stem="infer_batch_summary",
        payload={
            "artifact_type": "infer_batch_summary",
            "created_at_utc": now_utc(),
            "config_path": str(config_path),
            "data_source": str(config.get("data", {}).get("source", "synthetic")) if isinstance(config.get("data"), Mapping) else "synthetic",
            "model_name": model_name,
            "run_id": manifest.run_id,
            "manifest_path": str(run_dir / "manifest.json"),
            "batch_size": len(rows),
            "summary_csv_path": None if csv_path is None else str(csv_path),
            "final_phi_mean_min": min(objectives) if objectives else 0.0,
            "final_phi_mean_max": max(objectives) if objectives else 0.0,
            "final_phi_mean_avg": fmean(objectives) if objectives else 0.0,
            "records": rows,
        },
    )
    print(f"infer-batch summary written: {summary_path}")
    if csv_path is not None:
        print(f"infer-batch table written: {csv_path}")
    return 0


def cmd_search_recipe(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    dataset = build_dataset(config)
    if not dataset.runs:
        raise ValueError("search-recipe: synthetic dataset is empty")
    template = dataset.runs[0]
    ranges = search_ranges(config=config, fallback=template.recipe)
    trials = max(1, int(args.trials))

    config_path, config, out_dir, manifest, run_dir = _run_pipeline_compat(
        args=args,
        stages=["cleaning", "featurization", "preprocessing", "train", "inference"],
        override={
            "inference": {
                "mode": "optimize",
                "condition_ranges": {str(k): [float(v[0]), float(v[1])] for k, v in ranges.items()},
                "trials": int(trials),
                "seed": int(args.seed),
                "strategy": str(args.strategy).strip().lower(),
                "engine": str(getattr(args, "engine", "builtin")),
                "optuna_sampler": str(getattr(args, "optuna_sampler", "tpe")),
                "bo_candidates": max(6, int(args.bo_candidates)),
                "mfbo_pool_size": max(3, int(args.mfbo_pool_size)),
                "mfbo_top_k": max(1, int(args.mfbo_top_k)),
            }
        },
    )
    inference_artifacts = _artifact_index(manifest, "inference")
    if "inference_optimize_summary" not in inference_artifacts:
        raise ValueError("search-recipe pipeline output missing inference_optimize_summary artifact")
    optimize_summary = _read_json(inference_artifacts["inference_optimize_summary"])
    history_json = (
        _read_json(inference_artifacts["inference_optimize_history_json"])
        if inference_artifacts.get("inference_optimize_history_json") and inference_artifacts["inference_optimize_history_json"].exists()
        else {"history": []}
    )
    history = [dict(row) for row in history_json.get("history", [])] if isinstance(history_json.get("history"), list) else []
    best_entry = dict(optimize_summary.get("best_entry", {})) if isinstance(optimize_summary.get("best_entry"), Mapping) else {}
    csv_path = inference_artifacts.get("inference_optimize_history")
    summary_path = write_json_timestamped(
        out_dir=out_dir,
        stem="recipe_search_summary",
        payload={
            "artifact_type": "recipe_search_summary",
            "created_at_utc": now_utc(),
            "config_path": str(config_path),
            "data_source": str(config.get("data", {}).get("source", "synthetic")) if isinstance(config.get("data"), Mapping) else "synthetic",
            "run_id": manifest.run_id,
            "manifest_path": str(run_dir / "manifest.json"),
            "strategy": optimize_summary.get("strategy"),
            "requested_engine": optimize_summary.get("requested_engine"),
            "resolved_engine": optimize_summary.get("resolved_engine"),
            "objective_name": "objective",
            "objective_direction": "minimize",
            "num_trials": trials,
            "search_ranges": {key: [bounds[0], bounds[1]] for key, bounds in ranges.items()},
            "fidelity_counts": optimize_summary.get("fidelity_counts", {}),
            "fallback_reason": optimize_summary.get("fallback_reason"),
            "history_csv_path": None if csv_path is None else str(csv_path),
            "best_candidate": best_entry.get("conditions", {}),
            "best_objective": float(best_entry.get("objective", 0.0)),
            "objective_history": [float(row.get("objective", 0.0)) for row in history],
            "history": history,
        },
    )
    print(f"search-recipe summary written: {summary_path}")
    if csv_path is not None:
        print(f"search-recipe history written: {csv_path}")
    if best_entry:
        print(f"search-recipe best objective: {best_entry.get('objective')}")
    fallback_reason = optimize_summary.get("fallback_reason")
    if isinstance(fallback_reason, str) and fallback_reason:
        print(f"search-recipe fallback: {fallback_reason}")
    return 0


def cmd_sweep_pipelines(args: argparse.Namespace) -> int:
    config_path, config, out_dir, manifest, run_dir = _run_pipeline_compat(
        args=args,
        stages=["cleaning", "featurization", "preprocessing", "train", "inference"],
        override={},
    )
    leaderboard_root = run_dir / "leaderboard"
    rows: list[dict[str, Any]] = []
    for board_name in ("data_path", "model_path"):
        csv_path = leaderboard_root / board_name / "leaderboard.csv"
        if not csv_path.exists() or not csv_path.is_file():
            continue
        import csv

        with csv_path.open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                rows.append(
                    {
                        "leaderboard": board_name,
                        "name": str(row.get("name", "")),
                        "score": float(row.get("score", 0.0)),
                    }
                )
    table_path = write_csv_timestamped(
        out_dir=out_dir,
        stem="pipeline_sweep_table",
        fieldnames=["leaderboard", "name", "score"],
        rows=rows,
    )
    best_row = min(rows, key=lambda row: float(row["score"])) if rows else {"leaderboard": "", "name": "", "score": 0.0}
    summary_path = write_json_timestamped(
        out_dir=out_dir,
        stem="pipeline_sweep_summary",
        payload={
            "artifact_type": "pipeline_sweep_summary",
            "created_at_utc": now_utc(),
            "config_path": str(config_path),
            "data_source": str(config.get("data", {}).get("source", "synthetic")) if isinstance(config.get("data"), Mapping) else "synthetic",
            "run_id": manifest.run_id,
            "manifest_path": str(run_dir / "manifest.json"),
            "num_combinations": len(rows),
            "comparison_table_csv_path": str(table_path),
            "best_combo": best_row,
            "results": rows,
        },
    )
    print(f"sweep-pipelines summary written: {summary_path}")
    print(f"sweep-pipelines table written: {table_path}")
    return 0


def cmd_pipeline_run(args: argparse.Namespace) -> int:
    stages: str | list[str] | None
    if args.stages is None:
        stages = None
    else:
        stages = [part.strip() for part in str(args.stages).split(",") if part.strip()]
    manifest = run_stage_pipeline(config_path=args.config, selected_stages=stages)
    print(f"pipeline run_id: {manifest.run_id}")
    print(f"pipeline manifest: runs/{manifest.run_id}/manifest.json")
    print(f"pipeline stage order: {','.join(manifest.stage_order)}")
    for stage in manifest.stage_order:
        print(f"pipeline stage {stage}: {manifest.stage_status.get(stage, 'unknown')}")
    return 0
