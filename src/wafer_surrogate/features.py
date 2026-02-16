from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from wafer_surrogate.registries import Registry


FeatureMap = dict[str, float]
RawSample = Mapping[str, float]


class FeatureExtractor(Protocol):
    def extract(self, sample: RawSample) -> FeatureMap:
        ...


FeatureExtractorFactory = Callable[..., FeatureExtractor]
FEATURE_EXTRACTOR_REGISTRY = Registry[FeatureExtractorFactory]("feature_extractor")


def register_feature_extractor(name: str) -> Callable[[FeatureExtractorFactory], FeatureExtractorFactory]:
    return FEATURE_EXTRACTOR_REGISTRY.register(name)


def list_feature_extractors() -> list[str]:
    return FEATURE_EXTRACTOR_REGISTRY.list()


def make_feature_extractor(name: str, **kwargs: object) -> FeatureExtractor:
    extractor = FEATURE_EXTRACTOR_REGISTRY.create(name, **kwargs)
    if not hasattr(extractor, "extract"):
        raise TypeError(f"feature_extractor: '{name}' does not implement extract(...)")
    return extractor  # type: ignore[return-value]


@dataclass
class IdentityFeatureExtractor:
    prefix: str = "feat_"

    def extract(self, sample: RawSample) -> FeatureMap:
        return {f"{self.prefix}{key}": float(value) for key, value in sample.items()}


@register_feature_extractor("identity")
def _build_identity_feature_extractor(prefix: str = "feat_") -> FeatureExtractor:
    return IdentityFeatureExtractor(prefix=prefix)


@dataclass
class RecipeStatsFeatureExtractor:
    prefix: str = "stat_"

    def extract(self, sample: RawSample) -> FeatureMap:
        values = [float(value) for value in sample.values()]
        if not values:
            return {f"{self.prefix}count": 0.0}
        mean_value = sum(values) / float(len(values))
        variance = sum((value - mean_value) ** 2 for value in values) / float(len(values))
        return {
            f"{self.prefix}count": float(len(values)),
            f"{self.prefix}mean": mean_value,
            f"{self.prefix}std": math.sqrt(variance),
            f"{self.prefix}min": min(values),
            f"{self.prefix}max": max(values),
            f"{self.prefix}span": max(values) - min(values),
        }


@register_feature_extractor("recipe_stats")
def _build_recipe_stats_feature_extractor(prefix: str = "stat_") -> FeatureExtractor:
    return RecipeStatsFeatureExtractor(prefix=prefix)
