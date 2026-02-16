from __future__ import annotations

import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wafer_surrogate.config.loader import load_config
from wafer_surrogate.config.schema import PipelineRunConfig, build_pipeline_run_config
from wafer_surrogate.data.synthetic import SyntheticSDFDataset, generate_synthetic_sdf_dataset
from wafer_surrogate.pipeline.stages import (
    DataCleaningStage,
    FeaturizationStage,
    InferenceStage,
    PreprocessingStage,
    TrainStage,
)
from wafer_surrogate.pipeline.types import ArtifactRef, LeaderboardRecord, PipelineManifest, StageResult
from wafer_surrogate.pipeline.utils import ensure_stage_dirs, now_utc, sanitize_run_id, write_csv, write_json
from wafer_surrogate.viz.leaderboard import render_leaderboard_for_run


STAGE_IMPL = {
    "cleaning": DataCleaningStage,
    "featurization": FeaturizationStage,
    "preprocessing": PreprocessingStage,
    "train": TrainStage,
    "inference": InferenceStage,
}

STAGE_DEPENDENCIES = {
    "cleaning": [],
    "featurization": ["cleaning"],
    "preprocessing": ["cleaning", "featurization"],
    "train": ["preprocessing"],
    "inference": ["train", "preprocessing", "featurization", "cleaning"],
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[str(key)] = _deep_merge(dict(out[str(key)]), value)
        else:
            out[str(key)] = value
    return out


@dataclass
class PipelineRuntime:
    run_config: PipelineRunConfig
    run_dir: Path
    payload: dict[str, Any] = field(default_factory=dict)
    stage_cfg: dict[str, dict[str, Any]] = field(default_factory=dict)

    def stage_params(self, name: str) -> dict[str, Any]:
        return dict(self.stage_cfg.get(name, {}))


def _build_dataset(config: dict[str, Any]) -> SyntheticSDFDataset:
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("config[data] must be a mapping")
    source = str(data_cfg.get("source", "synthetic"))
    if source != "synthetic":
        raise ValueError(f"unsupported data.source='{source}' for pipeline run")
    return generate_synthetic_sdf_dataset(
        num_runs=int(data_cfg.get("num_runs", 1)),
        num_steps=int(data_cfg.get("num_steps", 6)),
        grid_size=int(data_cfg.get("grid_size", 24)),
        dt=float(data_cfg.get("dt", 0.1)),
        dimension=int(data_cfg.get("dimension", 2)),
        grid_depth=int(data_cfg.get("grid_depth", 16)),
    )


def _write_leaderboards(run_dir: Path, stage_results: dict[str, StageResult]) -> dict[str, list[LeaderboardRecord]]:
    leaderboards_dir = run_dir / "leaderboard"
    leaderboards_dir.mkdir(parents=True, exist_ok=True)

    data_rows: list[LeaderboardRecord] = []
    model_rows: list[LeaderboardRecord] = []

    train_result = stage_results.get("train")
    if train_result is not None:
        primary_metric = "student_mae" if "student_mae" in train_result.metrics else "mae"
        secondary_metric = "rollout_short_window_error" if "rollout_short_window_error" in train_result.metrics else "rmse"
        score = float(train_result.metrics.get(primary_metric, train_result.metrics.get("mae", 0.0)))
        secondary_score = float(train_result.metrics.get(secondary_metric, train_result.metrics.get("rmse", score)))
        data_rows.append(
            LeaderboardRecord(
                name="default_data_path",
                score=score,
                metrics=dict(train_result.metrics),
                metadata={
                    "source_stage": "train",
                    "primary_metric": primary_metric,
                    "secondary_metric": secondary_metric,
                    "secondary_score": secondary_score,
                },
            )
        )
        model_name = str(train_result.details.get("best_model", "model"))
        model_rows.append(
            LeaderboardRecord(
                name=model_name,
                score=score,
                metrics=dict(train_result.metrics),
                metadata={
                    "source_stage": "train",
                    "primary_metric": primary_metric,
                    "secondary_metric": secondary_metric,
                    "secondary_score": secondary_score,
                },
            )
        )

    if data_rows:
        write_csv(
            leaderboards_dir / "data_path" / "leaderboard.csv",
            [
                {
                    "name": row.name,
                    "score": row.score,
                    "primary_metric": str(row.metadata.get("primary_metric", "student_mae")),
                    "secondary_metric": str(row.metadata.get("secondary_metric", "rollout_short_window_error")),
                    "secondary_score": float(row.metadata.get("secondary_score", row.score)),
                    **{f"metric_{k}": v for k, v in row.metrics.items()},
                }
                for row in data_rows
            ],
            [
                "name",
                "score",
                "primary_metric",
                "secondary_metric",
                "secondary_score",
                *sorted({f"metric_{k}" for row in data_rows for k in row.metrics}),
            ],
        )

    if model_rows:
        write_csv(
            leaderboards_dir / "model_path" / "leaderboard.csv",
            [
                {
                    "name": row.name,
                    "score": row.score,
                    "primary_metric": str(row.metadata.get("primary_metric", "student_mae")),
                    "secondary_metric": str(row.metadata.get("secondary_metric", "rollout_short_window_error")),
                    "secondary_score": float(row.metadata.get("secondary_score", row.score)),
                    **{f"metric_{k}": v for k, v in row.metrics.items()},
                }
                for row in model_rows
            ],
            [
                "name",
                "score",
                "primary_metric",
                "secondary_metric",
                "secondary_score",
                *sorted({f"metric_{k}" for row in model_rows for k in row.metrics}),
            ],
        )

    return {
        "data_path": data_rows,
        "model_path": model_rows,
    }


def run_pipeline(
    *,
    config_path: str | Path,
    selected_stages: list[str] | str | None = None,
    config_override: Mapping[str, Any] | None = None,
) -> PipelineManifest:
    config = load_config(config_path)
    if isinstance(config_override, Mapping) and config_override:
        config = _deep_merge(dict(config), config_override)
    run_cfg = build_pipeline_run_config(config_path=config_path, payload=config, selected_stages=selected_stages)

    run_id = run_cfg.run_id or f"pipeline_{now_utc()}"
    safe_run_id = sanitize_run_id(run_id)
    run_dir = Path(run_cfg.output_root) / safe_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    runtime = PipelineRuntime(
        run_config=run_cfg,
        run_dir=run_dir,
        payload={"config": config, "dataset_raw": _build_dataset(config)},
        stage_cfg={stage.name: dict(stage.params) for stage in run_cfg.stages},
    )

    stage_status: dict[str, str] = {}
    stage_dependencies: dict[str, list[str]] = {}
    stage_inputs: dict[str, dict[str, str]] = {}
    stage_artifacts: dict[str, list[ArtifactRef]] = {}
    stage_metrics: dict[str, dict[str, float]] = {}
    stage_results: dict[str, StageResult] = {}
    warnings: list[str] = []

    for stage_cfg in run_cfg.stages:
        stage_dependencies[stage_cfg.name] = list(STAGE_DEPENDENCIES.get(stage_cfg.name, []))
        stage_inputs[stage_cfg.name] = dict(stage_cfg.external_inputs)

        stage_dirs = ensure_stage_dirs(run_dir, stage_cfg.name)
        write_json(stage_dirs["configuration"] / "stage_config.json", stage_cfg.to_dict())

        if not stage_cfg.enabled:
            stage_status[stage_cfg.name] = "skipped"
            stage_artifacts[stage_cfg.name] = []
            stage_metrics[stage_cfg.name] = {}
            write_json(
                stage_dirs["logs"] / "stage_result.json",
                StageResult(stage=stage_cfg.name, status="skipped").to_dict(),
            )
            continue
        if stage_cfg.name not in STAGE_IMPL:
            stage_status[stage_cfg.name] = "unknown_stage"
            stage_artifacts[stage_cfg.name] = []
            stage_metrics[stage_cfg.name] = {}
            write_json(
                stage_dirs["logs"] / "stage_result.json",
                StageResult(stage=stage_cfg.name, status="unknown_stage").to_dict(),
            )
            warnings.append(f"unknown stage requested: {stage_cfg.name}")
            continue

        stage_impl = STAGE_IMPL[stage_cfg.name]()
        result = stage_impl.run(runtime, stage_dirs)
        stage_results[stage_cfg.name] = result
        stage_status[stage_cfg.name] = result.status
        stage_artifacts[stage_cfg.name] = result.artifacts
        stage_metrics[stage_cfg.name] = dict(result.metrics)
        if isinstance(result.details, dict):
            detail_inputs = result.details.get("input_refs")
            if isinstance(detail_inputs, dict):
                merged = dict(stage_inputs.get(stage_cfg.name, {}))
                merged.update({str(k): str(v) for k, v in detail_inputs.items()})
                stage_inputs[stage_cfg.name] = merged
            detail_warnings = result.details.get("warnings")
            if isinstance(detail_warnings, list):
                warnings.extend(str(message) for message in detail_warnings)
            fallback_reason = result.details.get("fallback_reason")
            if isinstance(fallback_reason, str) and fallback_reason.strip():
                warnings.append(f"{stage_cfg.name}: {fallback_reason}")
        write_json(stage_dirs["logs"] / "stage_result.json", result.to_dict())

    leaderboards = _write_leaderboards(run_dir, stage_results)
    try:
        leaderboard_viz = render_leaderboard_for_run(run_dir)
        warnings.append(f"leaderboard_viz_manifest={leaderboard_viz.get('summary_path')}")
    except Exception as exc:
        warnings.append(f"leaderboard visualization skipped: {exc}")

    run_seed = None
    run_cfg_payload = config.get("run")
    if isinstance(run_cfg_payload, dict):
        run_seed = run_cfg_payload.get("seed")
    seed_info: dict[str, Any] = {}
    if run_seed is not None:
        seed_info["run_seed"] = int(run_seed)
    split_info = runtime.payload.get("train_split_info") if isinstance(runtime.payload.get("train_split_info"), dict) else {}

    manifest = PipelineManifest(
        run_id=safe_run_id,
        config_path=str(Path(config_path)),
        created_at_utc=now_utc(),
        schema_version="2",
        stage_order=[stage.name for stage in run_cfg.stages],
        stage_status=stage_status,
        stage_dependencies=stage_dependencies,
        stage_inputs=stage_inputs,
        stage_artifacts=stage_artifacts,
        stage_metrics=stage_metrics,
        runtime_env={
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        seed_info=seed_info,
        split_info=split_info if isinstance(split_info, dict) else {},
        warnings=warnings,
        leaderboards=leaderboards,
    )
    write_json(run_dir / "manifest.json", manifest.to_dict())
    return manifest
