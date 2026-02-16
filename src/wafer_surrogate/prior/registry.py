from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol

from wafer_surrogate.registries import Registry


class ShapePrior(Protocol):
    def sample_latent(self, num_samples: int = 1, seed: int | None = None) -> list[list[float]]:
        ...

    def score_latent(self, latent: Sequence[float]) -> float:
        ...


PriorFactory = Callable[..., ShapePrior]
PRIOR_REGISTRY = Registry[PriorFactory]("shape_prior")


class _ValidatedShapePrior:
    def __init__(self, inner: ShapePrior) -> None:
        self._inner = inner

    def sample_latent(self, num_samples: int = 1, seed: int | None = None) -> list[list[float]]:
        rows = self._inner.sample_latent(num_samples=num_samples, seed=seed)
        if not isinstance(rows, list):
            raise TypeError("shape_prior contract violation: sample_latent must return list[list[float]]")
        out: list[list[float]] = []
        for idx, row in enumerate(rows):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
                raise TypeError(f"shape_prior contract violation: sample_latent row {idx} must be a numeric sequence")
            out.append([float(v) for v in row])
        return out

    def score_latent(self, latent: Sequence[float]) -> float:
        if not isinstance(latent, Sequence) or isinstance(latent, (str, bytes, bytearray)):
            raise ValueError("shape_prior score_latent: latent must be a numeric sequence")
        score = self._inner.score_latent([float(v) for v in latent])
        if not isinstance(score, (int, float)):
            raise TypeError("shape_prior contract violation: score_latent must return float")
        out = float(score)
        if not math.isfinite(out):
            raise ValueError("shape_prior score_latent: returned non-finite value")
        return out


def register_shape_prior(name: str) -> Callable[[PriorFactory], PriorFactory]:
    return PRIOR_REGISTRY.register(name)


def list_shape_priors() -> list[str]:
    return PRIOR_REGISTRY.list()


def make_shape_prior(name: str, **kwargs: object) -> ShapePrior:
    prior = PRIOR_REGISTRY.create(name, **kwargs)
    if not hasattr(prior, "sample_latent") or not hasattr(prior, "score_latent"):
        raise TypeError(f"shape_prior: '{name}' must implement sample_latent(...) and score_latent(...)")
    return _ValidatedShapePrior(prior)  # type: ignore[return-value]
