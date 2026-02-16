from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from wafer_surrogate.prior import make_shape_prior


_PRIOR_ALIASES = {
    "gaussian": "gaussian_latent",
}


def build_prior(config: Mapping[str, Any]) -> tuple[str, dict[str, Any], Any]:
    prior_cfg = config.get("prior", {})
    if not isinstance(prior_cfg, Mapping):
        raise ValueError("config[prior] must be a table")
    requested = str(prior_cfg.get("name", "gaussian_latent"))
    prior_name = _PRIOR_ALIASES.get(requested, requested)
    kwargs: dict[str, Any] = {"latent_dim": int(prior_cfg.get("latent_dim", 1))}
    for key in ("mean", "std"):
        if key not in prior_cfg:
            continue
        value = prior_cfg[key]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            kwargs[key] = [float(v) for v in value]
        else:
            kwargs[key] = float(value)
    return prior_name, kwargs, make_shape_prior(prior_name, **kwargs)


def prior_preview(prior: Any) -> dict[str, Any]:
    sample = prior.sample_latent(num_samples=1, seed=0)[0]
    score = prior.score_latent(sample)
    return {
        "latent_dim": len(sample),
        "sample": [float(v) for v in sample],
        "sample_score": float(score),
    }
