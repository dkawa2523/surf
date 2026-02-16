from __future__ import annotations

import math
from collections.abc import Sequence

from wafer_surrogate.observation.plugins import BaselineSdfObservationModel
from wafer_surrogate.observation.registry import (
    OBSERVATION_MODEL_REGISTRY,
    ObservationModel,
    ObservationModelError,
    ShapeState,
    list_observation_models,
    make_observation_model,
    register_observation_model,
)


def _as_sequence(name: str, value: object) -> list[object]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ObservationModelError(f"{name} must be a sequence")
    return list(value)


def _safe_metric_key(name: str) -> str:
    text = str(name).strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "feature"


def compute_observation_feature_metrics(
    observation_model: ObservationModel,
    predicted_shapes: Sequence[ShapeState],
    reference_shapes: Sequence[ShapeState],
) -> dict[str, float]:
    pred = _as_sequence("predicted_shapes", predicted_shapes)
    ref = _as_sequence("reference_shapes", reference_shapes)
    if len(pred) != len(ref):
        raise ObservationModelError("predicted_shapes and reference_shapes length must match")
    if not pred:
        raise ObservationModelError("predicted_shapes/reference_shapes must be non-empty")

    abs_total = 0.0
    sq_total = 0.0
    dim: int | None = None
    abs_by_dim: list[float] = []
    sq_by_dim: list[float] = []
    feature_names: list[str] = []
    for idx, (pred_shape, ref_shape) in enumerate(zip(pred, ref)):
        y_pred = observation_model.project(pred_shape)  # type: ignore[arg-type]
        y_ref = observation_model.project(ref_shape)  # type: ignore[arg-type]
        if dim is None:
            dim = len(y_pred)
            if dim == 0:
                raise ObservationModelError("observation_model produced an empty feature vector")
            abs_by_dim = [0.0 for _ in range(dim)]
            sq_by_dim = [0.0 for _ in range(dim)]
            if hasattr(observation_model, "feature_names"):
                try:
                    feature_names = list(observation_model.feature_names())
                except Exception:
                    feature_names = []
            if len(feature_names) != dim:
                feature_names = [f"feature_{i:03d}" for i in range(dim)]
        if len(y_pred) != len(y_ref):
            raise ObservationModelError(f"feature dimension mismatch at pair {idx}")
        for dim_idx, (pred_value, ref_value) in enumerate(zip(y_pred, y_ref)):
            diff = float(pred_value) - float(ref_value)
            abs_total += abs(diff)
            sq_total += diff * diff
            abs_by_dim[dim_idx] += abs(diff)
            sq_by_dim[dim_idx] += diff * diff

    total = float(len(pred) * int(dim or 0))
    if total <= 0.0:
        raise ObservationModelError("no comparable projected features")
    metrics = {
        "obs_feature_mae": abs_total / total,
        "obs_feature_rmse": math.sqrt(sq_total / total),
        "obs_pairs_compared": float(len(pred)),
        "obs_feature_dim": float(dim or 0),
    }
    if dim is not None and dim > 0:
        for dim_idx in range(dim):
            count = float(len(pred))
            name = _safe_metric_key(feature_names[dim_idx])
            metrics[f"obs_{name}_mae"] = abs_by_dim[dim_idx] / count
            metrics[f"obs_{name}_rmse"] = math.sqrt(sq_by_dim[dim_idx] / count)
    return metrics


__all__ = [
    "BaselineSdfObservationModel",
    "OBSERVATION_MODEL_REGISTRY",
    "ObservationModel",
    "ObservationModelError",
    "ShapeState",
    "compute_observation_feature_metrics",
    "list_observation_models",
    "make_observation_model",
    "register_observation_model",
]
