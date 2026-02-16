from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:  # Optional dependency in P0.
    import numpy as _np
except Exception:  # pragma: no cover - environment dependent
    _np = None


def _require_numpy() -> Any:
    if _np is None:  # pragma: no cover - exercised only when numpy missing
        raise RuntimeError("numpy is required for wafer_surrogate.geometry utilities")
    return _np


def _normalize_spacing(ndim: int, spacing: float | Sequence[float]) -> tuple[float, ...]:
    if isinstance(spacing, Sequence) and not isinstance(spacing, (str, bytes)):
        values = tuple(float(value) for value in spacing)
        if len(values) != ndim:
            raise ValueError(f"spacing must have length {ndim}, got {len(values)}")
        return values
    return tuple(float(spacing) for _ in range(ndim))


def extract_narrow_band(phi: Any, band_width: float = 1.0) -> tuple[Any, Any]:
    """Return sparse narrow-band coordinates and SDF values around the zero level-set."""
    if band_width < 0:
        raise ValueError("band_width must be >= 0")

    np = _require_numpy()
    phi_arr = np.asarray(phi, dtype=float)
    mask = np.abs(phi_arr) <= float(band_width)
    coords = np.argwhere(mask)
    values = phi_arr[mask]
    return coords, values


def finite_diff_grad(phi: Any, spacing: float | Sequence[float] = 1.0) -> Any:
    """Compute central-difference gradient with first axis indexing gradient component."""
    np = _require_numpy()
    phi_arr = np.asarray(phi, dtype=float)
    if phi_arr.ndim < 1:
        raise ValueError("phi must have at least 1 dimension")

    spacing_seq = _normalize_spacing(phi_arr.ndim, spacing)
    edge_order = 2 if all(size >= 3 for size in phi_arr.shape) else 1
    grads = np.gradient(phi_arr, *spacing_seq, edge_order=edge_order)
    return np.stack(grads, axis=0)


def levelset_step(
    phi: Any,
    vn: Any,
    dt: float,
    spacing: float | Sequence[float] = 1.0,
) -> Any:
    """Apply one explicit level-set update: phi_next = phi - dt * vn * |grad(phi)|."""
    if dt <= 0:
        raise ValueError("dt must be > 0")

    np = _require_numpy()
    phi_arr = np.asarray(phi, dtype=float)
    vn_arr = np.asarray(vn, dtype=float)
    if phi_arr.shape != vn_arr.shape:
        raise ValueError(
            f"phi and vn must have the same shape, got {phi_arr.shape} and {vn_arr.shape}"
        )

    grad = finite_diff_grad(phi_arr, spacing=spacing)
    grad_norm = np.linalg.norm(grad, axis=0)
    return phi_arr - float(dt) * vn_arr * grad_norm


def reinit_sussman(
    phi: Any,
    *,
    iters: int = 8,
    dt: float = 0.3,
    spacing: float | Sequence[float] = 1.0,
) -> Any:
    """Apply simple Sussman reinitialization to enforce |grad(phi)| ~= 1."""
    if iters < 1:
        raise ValueError("iters must be >= 1")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    np = _require_numpy()
    phi0 = np.asarray(phi, dtype=float)
    sign0 = phi0 / np.sqrt((phi0 * phi0) + 1.0)
    phi_state = np.asarray(phi0, dtype=float)
    for _ in range(int(iters)):
        grad = finite_diff_grad(phi_state, spacing=spacing)
        grad_norm = np.linalg.norm(grad, axis=0)
        phi_state = phi_state - float(dt) * sign0 * (grad_norm - 1.0)
    return phi_state
