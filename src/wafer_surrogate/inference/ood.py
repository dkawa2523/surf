from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _sorted_feature_keys(recipes: Sequence[Mapping[str, float]]) -> list[str]:
    keys = set()
    for recipe in recipes:
        for key in recipe:
            keys.add(str(key))
    return sorted(keys)


def _recipe_vector(recipe: Mapping[str, float], feature_keys: Sequence[str]) -> list[float]:
    return [float(recipe.get(key, 0.0)) for key in feature_keys]


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    dim = len(vectors[0])
    out = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec):
            out[i] += float(value)
    inv_n = 1.0 / float(len(vectors))
    return [value * inv_n for value in out]


def _scale_vector(vectors: Sequence[Sequence[float]], center: Sequence[float]) -> list[float]:
    dim = len(center)
    if len(vectors) < 2:
        return [1.0] * dim

    out = [0.0] * dim
    inv_n = 1.0 / float(len(vectors))
    for vec in vectors:
        for i, value in enumerate(vec):
            diff = float(value) - float(center[i])
            out[i] += diff * diff
    return [max(1e-6, math.sqrt(value * inv_n)) for value in out]


def _normalized_distance(
    lhs: Sequence[float],
    rhs: Sequence[float],
    scale: Sequence[float],
) -> float:
    if not lhs:
        return 0.0
    sq = 0.0
    for i, lhs_value in enumerate(lhs):
        denom = float(scale[i]) if float(scale[i]) > 0.0 else 1.0
        diff = (float(lhs_value) - float(rhs[i])) / denom
        sq += diff * diff
    return math.sqrt(sq / float(len(lhs)))


def assess_ood(
    query_conditions: Mapping[str, float],
    reference_conditions: Sequence[Mapping[str, float]],
    threshold: float = 3.0,
) -> dict[str, object]:
    refs = [
        {str(key): float(value) for key, value in conditions.items()}
        for conditions in reference_conditions
    ]
    keys = _sorted_feature_keys(refs + [{str(k): float(v) for k, v in query_conditions.items()}])

    if not refs:
        return {
            "status": "insufficient_reference",
            "in_domain": False,
            "distance": None,
            "threshold": float(threshold),
            "num_reference": 0,
            "feature_keys": keys,
        }

    query = _recipe_vector(query_conditions, keys)
    ref_vectors = [_recipe_vector(recipe, keys) for recipe in refs]
    center = _mean_vector(ref_vectors)
    scale = _scale_vector(ref_vectors, center)

    mahalanobis_distance = _normalized_distance(query, center, scale)
    knn_distance = min(_normalized_distance(query, ref_vec, scale) for ref_vec in ref_vectors)
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
