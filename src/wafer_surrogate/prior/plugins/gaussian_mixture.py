from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from wafer_surrogate.prior.registry import ShapePrior, register_shape_prior


def _as_vector(values: Sequence[float], *, latent_dim: int, name: str) -> list[float]:
    out = [float(v) for v in values]
    if len(out) != latent_dim:
        raise ValueError(f"{name} length must match latent_dim")
    return out


def _normalize_weights(values: Sequence[float]) -> list[float]:
    out = [max(0.0, float(v)) for v in values]
    total = sum(out)
    if total <= 0.0:
        raise ValueError("gaussian_mixture weights must contain at least one positive value")
    return [value / total for value in out]


@dataclass(frozen=True)
class GaussianMixtureLatentPrior:
    latent_dim: int = 1
    weights: Sequence[float] = (1.0,)
    means: Sequence[Sequence[float]] = ((0.0,),)
    stds: Sequence[Sequence[float]] = ((1.0,),)

    def _params(self) -> tuple[list[float], list[list[float]], list[list[float]]]:
        dim = max(1, int(self.latent_dim))
        weights = _normalize_weights(self.weights)
        means = [_as_vector(row, latent_dim=dim, name="means") for row in self.means]
        stds = [_as_vector(row, latent_dim=dim, name="stds") for row in self.stds]
        if not (len(weights) == len(means) == len(stds)):
            raise ValueError("gaussian_mixture weights/means/stds component counts must match")
        for row in stds:
            if any(float(value) <= 0.0 for value in row):
                raise ValueError("gaussian_mixture stds must be > 0")
        return weights, means, stds

    def _pick_component(self, rng: random.Random, weights: Sequence[float]) -> int:
        u = rng.random()
        acc = 0.0
        for idx, weight in enumerate(weights):
            acc += float(weight)
            if u <= acc:
                return idx
        return len(weights) - 1

    def sample_latent(self, num_samples: int = 1, seed: int | None = None) -> list[list[float]]:
        if int(num_samples) < 1:
            raise ValueError("num_samples must be >= 1")
        weights, means, stds = self._params()
        rng = random.Random(seed)
        out: list[list[float]] = []
        for _ in range(int(num_samples)):
            comp = self._pick_component(rng, weights)
            row = [float(rng.gauss(mu, sd)) for mu, sd in zip(means[comp], stds[comp])]
            out.append(row)
        return out

    def score_latent(self, latent: Sequence[float]) -> float:
        weights, means, stds = self._params()
        z = _as_vector(latent, latent_dim=max(1, int(self.latent_dim)), name="latent")
        log_terms: list[float] = []
        for weight, mu, sd in zip(weights, means, stds):
            log_prob = math.log(max(1e-12, float(weight)))
            for value, mean, sigma in zip(z, mu, sd):
                std = float(sigma)
                norm = (float(value) - float(mean)) / std
                log_prob += -0.5 * (norm * norm) - math.log(std * math.sqrt(2.0 * math.pi))
            log_terms.append(log_prob)
        max_log = max(log_terms)
        lse = max_log + math.log(sum(math.exp(term - max_log) for term in log_terms))
        return float(-lse)


@register_shape_prior("gaussian_mixture")
def _build_gaussian_mixture_prior(
    latent_dim: int = 1,
    weights: Sequence[float] = (1.0,),
    means: Sequence[Sequence[float]] = ((0.0,),),
    stds: Sequence[Sequence[float]] = ((1.0,),),
) -> ShapePrior:
    return GaussianMixtureLatentPrior(
        latent_dim=max(1, int(latent_dim)),
        weights=weights,
        means=means,
        stds=stds,
    )
