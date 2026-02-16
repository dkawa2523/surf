from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from wafer_surrogate.inference.ood import assess_ood


def _sorted_keys(rows: Sequence[Mapping[str, float]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(str(k) for k in row)
    return sorted(keys)


def _vector(row: Mapping[str, float], keys: Sequence[str]) -> list[float]:
    return [float(row.get(key, 0.0)) for key in keys]


def _mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    dim = len(vectors[0])
    out = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec):
            out[i] += float(value)
    inv_n = 1.0 / float(len(vectors))
    return [value * inv_n for value in out]


def _scale(vectors: Sequence[Sequence[float]], center: Sequence[float]) -> list[float]:
    if len(vectors) < 2:
        return [1.0] * len(center)
    out = [0.0] * len(center)
    inv_n = 1.0 / float(len(vectors))
    for vec in vectors:
        for i, value in enumerate(vec):
            diff = float(value) - float(center[i])
            out[i] += diff * diff
    return [max(1e-6, math.sqrt(value * inv_n)) for value in out]


def _distance(lhs: Sequence[float], rhs: Sequence[float], scale: Sequence[float]) -> float:
    if not lhs:
        return 0.0
    sq = 0.0
    for i, lhs_value in enumerate(lhs):
        denom = float(scale[i]) if float(scale[i]) > 0.0 else 1.0
        diff = (float(lhs_value) - float(rhs[i])) / denom
        sq += diff * diff
    return math.sqrt(sq / float(len(lhs)))


def assess_feature_ood(
    *,
    query_features: Mapping[str, float],
    reference_features: Sequence[Mapping[str, float]],
    threshold: float = 3.0,
) -> dict[str, Any]:
    refs = [{str(k): float(v) for k, v in row.items()} for row in reference_features]
    keys = _sorted_keys(refs + [{str(k): float(v) for k, v in query_features.items()}])
    if not refs:
        return {
            "status": "insufficient_reference",
            "in_domain": False,
            "distance": None,
            "centroid_distance": None,
            "threshold": float(threshold),
            "num_reference": 0,
            "feature_keys": keys,
        }

    query_vec = _vector({str(k): float(v) for k, v in query_features.items()}, keys)
    ref_vecs = [_vector(row, keys) for row in refs]
    center = _mean(ref_vecs)
    scales = _scale(ref_vecs, center)
    mahalanobis_distance = _distance(query_vec, center, scales)
    knn_distance = min(_distance(query_vec, ref_vec, scales) for ref_vec in ref_vecs)
    combined_distance = max(float(mahalanobis_distance), float(knn_distance))
    in_domain = combined_distance <= float(threshold)
    return {
        "status": "in_domain" if in_domain else "out_of_domain",
        "in_domain": in_domain,
        "distance": combined_distance,
        "centroid_distance": mahalanobis_distance,
        "mahalanobis_distance": mahalanobis_distance,
        "knn_distance": knn_distance,
        "threshold": float(threshold),
        "num_reference": len(refs),
        "feature_keys": keys,
    }


def assess_dual_ood(
    *,
    query_conditions: Mapping[str, float],
    reference_conditions: Sequence[Mapping[str, float]],
    query_features: Mapping[str, float] | None,
    reference_features: Sequence[Mapping[str, float]] | None,
    threshold: float = 3.0,
    threshold_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    def _profile_threshold(bucket: str, default: float) -> float:
        if not isinstance(threshold_profile, Mapping):
            return float(default)
        block = threshold_profile.get(bucket)
        if isinstance(block, Mapping) and isinstance(block.get("threshold"), (int, float)):
            return float(block["threshold"])
        return float(default)

    condition_threshold = _profile_threshold("condition", float(threshold))
    feature_threshold = _profile_threshold("feature", float(threshold))
    condition = assess_ood(
        query_conditions=query_conditions,
        reference_conditions=reference_conditions,
        threshold=float(condition_threshold),
    )
    feature: dict[str, Any] | None = None
    if query_features is not None and reference_features is not None:
        feature = assess_feature_ood(
            query_features=query_features,
            reference_features=reference_features,
            threshold=float(feature_threshold),
        )

    is_in = bool(condition.get("in_domain", False)) and (
        True if feature is None else bool(feature.get("in_domain", False))
    )
    condition_score = condition.get("distance")
    feature_score = feature.get("distance") if isinstance(feature, Mapping) else None
    return {
        "status": "in_domain" if is_in else "out_of_domain",
        "in_domain": is_in,
        "combined_status": "in_domain" if is_in else "out_of_domain",
        "condition_score": condition_score,
        "feature_score": feature_score,
        "condition": condition,
        "feature": feature,
        "threshold": float(threshold),
        "condition_threshold": float(condition_threshold),
        "feature_threshold": float(feature_threshold),
    }
