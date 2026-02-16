from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from wafer_surrogate.prior.registry import ShapePrior, register_shape_prior


def _as_vector(
    value: float | Sequence[float] | None,
    *,
    latent_dim: int,
    default: float,
    name: str,
) -> list[float]:
    if latent_dim < 1:
        raise ValueError("latent_dim must be >= 1")
    if value is None:
        return [float(default) for _ in range(latent_dim)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value) for _ in range(latent_dim)]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a scalar or sequence")
    out = [float(v) for v in value]
    if len(out) != latent_dim:
        raise ValueError(f"{name} length must match latent_dim")
    return out


@dataclass(frozen=True)
class GaussianLatentPrior:
    latent_dim: int = 1
    mean: float | Sequence[float] | None = None
    std: float | Sequence[float] = 1.0

    def _params(self) -> tuple[list[float], list[float]]:
        mu = _as_vector(self.mean, latent_dim=int(self.latent_dim), default=0.0, name="mean")
        sigma = _as_vector(self.std, latent_dim=int(self.latent_dim), default=1.0, name="std")
        if any(s <= 0.0 for s in sigma):
            raise ValueError("std must be > 0")
        return mu, sigma

    def sample_latent(self, num_samples: int = 1, seed: int | None = None) -> list[list[float]]:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        mu, sigma = self._params()
        rng = random.Random(seed)
        return [
            [float(rng.gauss(m, s)) for m, s in zip(mu, sigma)]
            for _ in range(int(num_samples))
        ]

    def score_latent(self, latent: Sequence[float]) -> float:
        mu, sigma = self._params()
        if len(latent) != len(mu):
            raise ValueError("latent length must match latent_dim")
        total = 0.0
        for value, mean, std in zip(latent, mu, sigma):
            z = (float(value) - mean) / std
            total += 0.5 * (z * z) + math.log(std * math.sqrt(2.0 * math.pi))
        return float(total)


@register_shape_prior("gaussian_latent")
def _build_gaussian_latent_prior(
    latent_dim: int = 1,
    mean: float | Sequence[float] | None = None,
    std: float | Sequence[float] = 1.0,
) -> ShapePrior:
    return GaussianLatentPrior(latent_dim=latent_dim, mean=mean, std=std)
