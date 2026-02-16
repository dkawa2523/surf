from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from wafer_surrogate.observation import ObservationModel, ShapeState

try:  # Optional dependency in P2.
    import torch as _torch
except Exception:  # pragma: no cover - environment dependent
    _torch = None

try:  # Optional dependency in P2.
    from sbi.inference import SNPE as _SNPE
    from sbi.utils import BoxUniform as _BoxUniform
except Exception:  # pragma: no cover - environment dependent
    _SNPE = None
    _BoxUniform = None


PredictFeaturesFn = Callable[[Sequence[float]], Sequence[float]]
SimulateShapeFn = Callable[[Sequence[float]], ShapeState]


class CalibrationError(ValueError):
    """Invalid inputs/outputs for latent z MAP calibration."""


class OptionalDependencyUnavailable(RuntimeError):
    """Missing optional dependencies required for SBI posterior estimation."""


@dataclass(frozen=True)
class MapCalibrationResult:
    z_map: list[float]
    y_pred: list[float]
    objective: float
    feature_loss: float
    prior_loss: float
    grad_norm: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class SbiPosteriorEstimator:
    posterior: Any
    latent_dim: int
    observation_dim: int
    num_simulations: int
    prior_low: list[float]
    prior_high: list[float]
    device: str


def _vec(name: str, value: object) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CalibrationError(f"{name} must be a sequence")
    out = [float(v) for v in value]
    if not out:
        raise CalibrationError(f"{name} must contain at least one element")
    if any(not math.isfinite(v) for v in out):
        raise CalibrationError(f"{name} must contain finite values")
    return out


def _matrix(name: str, value: object) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CalibrationError(f"{name} must be a sequence of sequences")
    out: list[list[float]] = []
    width: int | None = None
    for idx, row in enumerate(value):
        row_vec = _vec(f"{name}[{idx}]", row)
        if width is None:
            width = len(row_vec)
        elif len(row_vec) != width:
            raise CalibrationError(f"{name} rows must have equal length")
        out.append(row_vec)
    if not out:
        raise CalibrationError(f"{name} must contain at least one row")
    return out


def _broadcast(value: float | Sequence[float] | None, *, n: int, default: float, name: str) -> list[float]:
    if value is None:
        return [default for _ in range(n)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value) for _ in range(n)]
    out = _vec(name, value)
    if len(out) != n:
        raise CalibrationError(f"{name} length must match required dimension")
    return out


def _require_sbi_backend() -> tuple[Any, Any, Any]:
    if _torch is None or _SNPE is None or _BoxUniform is None:
        missing: list[str] = []
        if _torch is None:
            missing.append("torch")
        if _SNPE is None or _BoxUniform is None:
            missing.append("sbi")
        raise OptionalDependencyUnavailable(
            "SBI posterior estimation requires optional dependency decision (DECISION-P1-001) "
            f"and installed packages: {', '.join(missing)}"
        )
    return _torch, _SNPE, _BoxUniform


def _objective(
    z: Sequence[float],
    *,
    target_y: Sequence[float],
    predict_features: PredictFeaturesFn,
    feature_weights: Sequence[float],
    prior_mean: Sequence[float],
    prior_std: Sequence[float],
) -> tuple[float, float, float, list[float]]:
    y_pred = _vec("predict_features(z)", predict_features(z))
    if len(y_pred) != len(target_y):
        raise CalibrationError("predict_features(z) output length must match target_y")
    feat = sum(
        0.5 * (((pred - obs) * w) ** 2)
        for pred, obs, w in zip(y_pred, target_y, feature_weights)
    )
    prior = sum(
        0.5 * (((value - mu) / sigma) ** 2)
        for value, mu, sigma in zip(z, prior_mean, prior_std)
    )
    return feat + prior, feat, prior, y_pred


def _finite_diff_grad(z: Sequence[float], fn: Callable[[Sequence[float]], float], eps: float) -> list[float]:
    out: list[float] = []
    for idx, value in enumerate(z):
        h = eps * max(1.0, abs(float(value)))
        plus = list(z)
        minus = list(z)
        plus[idx] += h
        minus[idx] -= h
        out.append((fn(plus) - fn(minus)) / (2.0 * h))
    return out


def calibrate_latent_map(
    predict_features: PredictFeaturesFn,
    target_y: Sequence[float],
    z_init: Sequence[float],
    *,
    prior_mean: Sequence[float] | None = None,
    prior_std: float | Sequence[float] = 1.0,
    feature_weights: Sequence[float] | None = None,
    learning_rate: float = 0.1,
    max_iters: int = 200,
    tol: float = 1e-7,
    fd_eps: float = 1e-4,
    min_learning_rate: float = 1e-8,
) -> MapCalibrationResult:
    """Estimate latent z by MAP while keeping surrogate/model weights fixed."""
    if learning_rate <= 0 or max_iters < 1 or tol <= 0 or fd_eps <= 0 or min_learning_rate <= 0:
        raise CalibrationError("invalid optimizer hyperparameters")

    y = _vec("target_y", target_y)
    z = _vec("z_init", z_init)
    mu = _broadcast(prior_mean, n=len(z), default=0.0, name="prior_mean")
    sigma = _broadcast(prior_std, n=len(z), default=1.0, name="prior_std")
    if any(s <= 0 for s in sigma):
        raise CalibrationError("prior_std must be > 0")
    w = _broadcast(feature_weights, n=len(y), default=1.0, name="feature_weights")

    def total(zv: Sequence[float]) -> float:
        return _objective(
            zv,
            target_y=y,
            predict_features=predict_features,
            feature_weights=w,
            prior_mean=mu,
            prior_std=sigma,
        )[0]

    obj, feat, prior, y_pred = _objective(
        z,
        target_y=y,
        predict_features=predict_features,
        feature_weights=w,
        prior_mean=mu,
        prior_std=sigma,
    )
    converged = False
    grad_norm = float("inf")
    steps = 0

    for step in range(1, max_iters + 1):
        grad = _finite_diff_grad(z, total, fd_eps)
        grad_norm = math.sqrt(sum(g * g for g in grad))
        steps = step
        if grad_norm <= tol:
            converged = True
            break

        step_size = learning_rate
        accepted = False
        while step_size >= min_learning_rate:
            trial = [v - step_size * g for v, g in zip(z, grad)]
            trial_obj, trial_feat, trial_prior, trial_pred = _objective(
                trial,
                target_y=y,
                predict_features=predict_features,
                feature_weights=w,
                prior_mean=mu,
                prior_std=sigma,
            )
            if trial_obj <= obj:
                improvement = obj - trial_obj
                z, obj, feat, prior, y_pred = trial, trial_obj, trial_feat, trial_prior, trial_pred
                accepted = True
                if improvement <= tol * max(1.0, abs(obj)):
                    converged = True
                break
            step_size *= 0.5

        if not accepted or converged:
            break

    return MapCalibrationResult(
        z_map=[float(v) for v in z],
        y_pred=[float(v) for v in y_pred],
        objective=float(obj),
        feature_loss=float(feat),
        prior_loss=float(prior),
        grad_norm=float(grad_norm),
        iterations=int(steps),
        converged=bool(converged),
    )


def calibrate_latent_map_with_observation(
    simulate_shape: SimulateShapeFn,
    observation_model: ObservationModel,
    target_y: Sequence[float],
    z_init: Sequence[float],
    **kwargs: object,
) -> MapCalibrationResult:
    """MAP calibration helper using an explicit shape->observation projector."""

    def _predict_features(z: Sequence[float]) -> list[float]:
        shape = simulate_shape(z)
        return observation_model.project(shape)

    return calibrate_latent_map(
        predict_features=_predict_features,
        target_y=target_y,
        z_init=z_init,
        **kwargs,
    )


def train_latent_posterior_sbi(
    latent_samples: Sequence[Sequence[float]],
    observations: Sequence[Sequence[float]],
    *,
    density_estimator: str = "maf",
    prior_padding_ratio: float = 0.1,
    training_batch_size: int = 64,
    max_num_epochs: int = 200,
    device: str = "cpu",
) -> SbiPosteriorEstimator:
    """Train SBI posterior p(z|y,c,geom) from synthetic (z, observation) pairs."""
    if prior_padding_ratio < 0:
        raise CalibrationError("prior_padding_ratio must be >= 0")
    if training_batch_size < 1 or max_num_epochs < 1:
        raise CalibrationError("training_batch_size and max_num_epochs must be >= 1")

    z_samples = _matrix("latent_samples", latent_samples)
    y_samples = _matrix("observations", observations)
    if len(z_samples) != len(y_samples):
        raise CalibrationError("latent_samples and observations length must match")
    if len(z_samples) < 2:
        raise CalibrationError("at least two simulation pairs are required")

    latent_dim = len(z_samples[0])
    observation_dim = len(y_samples[0])
    torch, SNPE, BoxUniform = _require_sbi_backend()

    prior_low: list[float] = []
    prior_high: list[float] = []
    for dim in range(latent_dim):
        dim_values = [sample[dim] for sample in z_samples]
        dim_min = min(dim_values)
        dim_max = max(dim_values)
        span = max(1e-6, dim_max - dim_min)
        pad = max(1e-6, span * float(prior_padding_ratio))
        prior_low.append(dim_min - pad)
        prior_high.append(dim_max + pad)

    theta = torch.tensor(z_samples, dtype=torch.float32, device=device)
    x = torch.tensor(y_samples, dtype=torch.float32, device=device)
    prior = BoxUniform(
        low=torch.tensor(prior_low, dtype=torch.float32, device=device),
        high=torch.tensor(prior_high, dtype=torch.float32, device=device),
    )

    try:
        inference = SNPE(prior=prior, density_estimator=density_estimator, device=device)
    except TypeError:
        inference = SNPE(prior=prior, density_estimator=density_estimator)
    inference = inference.append_simulations(theta, x)

    try:
        density_estimator_obj = inference.train(
            training_batch_size=int(training_batch_size),
            max_num_epochs=int(max_num_epochs),
            show_train_summary=False,
        )
    except TypeError:
        density_estimator_obj = inference.train(
            training_batch_size=int(training_batch_size),
            max_num_epochs=int(max_num_epochs),
        )
    posterior = inference.build_posterior(density_estimator_obj)
    return SbiPosteriorEstimator(
        posterior=posterior,
        latent_dim=int(latent_dim),
        observation_dim=int(observation_dim),
        num_simulations=int(len(z_samples)),
        prior_low=[float(v) for v in prior_low],
        prior_high=[float(v) for v in prior_high],
        device=str(device),
    )


def sample_latent_posterior_sbi(
    estimator: SbiPosteriorEstimator,
    observation: Sequence[float],
    *,
    num_samples: int = 1,
    seed: int | None = None,
) -> list[list[float]]:
    """Sample latent z candidates from a trained SBI posterior."""
    if num_samples < 1:
        raise CalibrationError("num_samples must be >= 1")
    obs = _vec("observation", observation)
    if len(obs) != estimator.observation_dim:
        raise CalibrationError("observation length must match trained observation_dim")

    torch, _, _ = _require_sbi_backend()
    if seed is not None:
        torch.manual_seed(int(seed))

    x = torch.tensor(obs, dtype=torch.float32, device=estimator.device).reshape(1, -1)
    samples = estimator.posterior.sample((int(num_samples),), x=x)
    if hasattr(samples, "detach"):
        raw = samples.detach().cpu().tolist()
    else:
        raw = list(samples)

    if not isinstance(raw, list) or not raw:
        raise CalibrationError("posterior.sample(...) returned no samples")
    if isinstance(raw[0], list) and raw[0] and isinstance(raw[0][0], list):
        if len(raw[0]) != 1:
            raise CalibrationError("unexpected posterior sample shape")
        raw = [row[0] for row in raw]

    sampled: list[list[float]] = []
    for idx, row in enumerate(raw):
        z = _vec(f"sample[{idx}]", row)
        if len(z) != estimator.latent_dim:
            raise CalibrationError("sampled latent dimension mismatch")
        sampled.append([float(v) for v in z])
    return sampled
