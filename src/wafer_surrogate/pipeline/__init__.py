"""Composable pipeline execution for cleaning/feature/preprocess/train/inference."""

from wafer_surrogate.pipeline.types import (
    ArtifactRef,
    LeaderboardRecord,
    PipelineManifest,
    ReconstructionBundle,
    StageConfig,
    StageResult,
)


def run_pipeline(*args: object, **kwargs: object) -> PipelineManifest:
    # Lazy import prevents circular import between config.schema and pipeline.orchestrator.
    from wafer_surrogate.pipeline.orchestrator import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)

__all__ = [
    "ArtifactRef",
    "LeaderboardRecord",
    "PipelineManifest",
    "ReconstructionBundle",
    "StageConfig",
    "StageResult",
    "run_pipeline",
]
