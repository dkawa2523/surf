from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    external_inputs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "enabled": bool(self.enabled),
            "external_inputs": dict(self.external_inputs),
        }


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    path: str
    kind: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "path": self.path,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class LeaderboardRecord:
    name: str
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": float(self.score),
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReconstructionBundle:
    id: str
    payload_path: str
    target_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    row_refs: list[dict[str, Any]] = field(default_factory=list)
    inverse_mapping: dict[str, Any] = field(default_factory=dict)
    post_inference_hooks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "payload_path": self.payload_path,
            "target_path": self.target_path,
            "metrics": dict(self.metrics),
            "row_refs": [dict(row) for row in self.row_refs],
            "inverse_mapping": dict(self.inverse_mapping),
            "post_inference_hooks": list(self.post_inference_hooks),
        }


@dataclass
class StageResult:
    stage: str
    status: str
    artifacts: list[ArtifactRef] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metrics": dict(self.metrics),
            "details": dict(self.details),
        }


@dataclass
class PipelineManifest:
    run_id: str
    config_path: str
    created_at_utc: str
    stage_order: list[str]
    stage_status: dict[str, str]
    stage_dependencies: dict[str, list[str]]
    stage_inputs: dict[str, dict[str, str]]
    stage_artifacts: dict[str, list[ArtifactRef]]
    stage_metrics: dict[str, dict[str, float]]
    schema_version: str = "2"
    runtime_env: dict[str, Any] = field(default_factory=dict)
    seed_info: dict[str, Any] = field(default_factory=dict)
    split_info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    leaderboards: dict[str, list[LeaderboardRecord]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_path": self.config_path,
            "created_at_utc": self.created_at_utc,
            "schema_version": self.schema_version,
            "stage_order": list(self.stage_order),
            "stage_status": dict(self.stage_status),
            "stage_dependencies": {
                stage: list(deps)
                for stage, deps in self.stage_dependencies.items()
            },
            "stage_inputs": {
                stage: dict(inputs)
                for stage, inputs in self.stage_inputs.items()
            },
            "stage_artifacts": {
                stage: [ref.to_dict() for ref in refs]
                for stage, refs in self.stage_artifacts.items()
            },
            "stage_metrics": {stage: dict(metrics) for stage, metrics in self.stage_metrics.items()},
            "runtime_env": dict(self.runtime_env),
            "seed_info": dict(self.seed_info),
            "split_info": dict(self.split_info),
            "warnings": list(self.warnings),
            "leaderboards": {
                key: [row.to_dict() for row in rows]
                for key, rows in self.leaderboards.items()
            },
        }
