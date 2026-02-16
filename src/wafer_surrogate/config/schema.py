from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wafer_surrogate.pipeline.types import StageConfig


DEFAULT_STAGE_ORDER = ["cleaning", "featurization", "preprocessing", "train", "inference"]


@dataclass(frozen=True)
class PipelineRunConfig:
    config_path: Path
    run_id: str | None
    output_root: str
    stages: list[StageConfig]


def _normalize_stage_list(value: object) -> list[str]:
    if value is None:
        return list(DEFAULT_STAGE_ORDER)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out = [str(item).strip() for item in value if str(item).strip()]
        return out or list(DEFAULT_STAGE_ORDER)
    return list(DEFAULT_STAGE_ORDER)


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def build_pipeline_run_config(
    *,
    config_path: str | Path,
    payload: Mapping[str, Any],
    selected_stages: Sequence[str] | str | None = None,
) -> PipelineRunConfig:
    pipe_cfg = _mapping(payload.get("pipeline"))
    stage_names = _normalize_stage_list(
        selected_stages if selected_stages is not None else pipe_cfg.get("stages")
    )

    external_inputs_cfg = _mapping(pipe_cfg.get("external_inputs"))
    stage_overrides = _mapping(pipe_cfg.get("stages_config"))

    stages: list[StageConfig] = []
    for stage_name in stage_names:
        stage_cfg = _mapping(payload.get(stage_name))
        override_cfg = _mapping(stage_overrides.get(stage_name))
        merged_params = {str(k): v for k, v in stage_cfg.items()}
        merged_params.update({str(k): v for k, v in override_cfg.items()})

        enabled = bool(merged_params.pop("enabled", True))
        ext_inputs = _mapping(external_inputs_cfg.get(stage_name))
        stages.append(
            StageConfig(
                name=stage_name,
                params=merged_params,
                enabled=enabled,
                external_inputs={str(k): str(v) for k, v in ext_inputs.items()},
            )
        )

    run_cfg = _mapping(payload.get("run"))
    run_id_raw = pipe_cfg.get("run_id")
    if run_id_raw is None:
        run_id_raw = run_cfg.get("run_id")

    return PipelineRunConfig(
        config_path=Path(config_path),
        run_id=None if run_id_raw is None else str(run_id_raw),
        output_root=str(run_cfg.get("output_dir", "runs")),
        stages=stages,
    )
