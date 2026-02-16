from __future__ import annotations

import math
import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from wafer_surrogate.geometry import finite_diff_grad
from wafer_surrogate.registries import Registry

try:  # Optional dependency in P0/P1.
    import numpy as _np
except Exception:  # pragma: no cover - environment dependent
    _np = None


FeatureMap = Mapping[str, float]
ConditionMap = Mapping[str, float]


def _require_numpy() -> Any:
    if _np is None:  # pragma: no cover - exercised only when numpy missing
        raise RuntimeError("numpy is required for this model")
    return _np


class Model(Protocol):
    def fit(self, features: Sequence[FeatureMap], targets: Sequence[float]) -> None:
        ...

    def predict(self, features: FeatureMap) -> float:
        ...

    def predict_vn(
        self,
        phi: Any,
        conditions: ConditionMap,
        step_index: int = 0,
    ) -> float | Any:
        ...


ModelFactory = Callable[..., Model]
MODEL_REGISTRY = Registry[ModelFactory]("model")


def _load_deprecated_aliases() -> dict[str, dict[str, str]]:
    aliases_path = Path(__file__).with_name("deprecated_aliases.json")
    try:
        with aliases_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, Mapping):
            return {}
        out: dict[str, dict[str, str]] = {}
        for raw_key, raw_value in payload.items():
            if not isinstance(raw_value, Mapping):
                continue
            name = str(raw_key).strip()
            target = str(raw_value.get("target", "")).strip()
            if not name or not target:
                continue
            out[name] = {
                "target": target,
                "remove_after": str(raw_value.get("remove_after", "")).strip(),
                "reason": str(raw_value.get("reason", "")).strip(),
            }
        return out
    except Exception:
        return {}


_DEPRECATED_MODEL_ALIASES = _load_deprecated_aliases()


def _format_alias_warning(
    *,
    requested_name: str,
    resolved_name: str,
    alias_meta: Mapping[str, str],
) -> str:
    remove_after = str(alias_meta.get("remove_after", "")).strip()
    reason = str(alias_meta.get("reason", "")).strip()
    parts = [f"model '{requested_name}' is a deprecated alias for '{resolved_name}'."]
    if reason:
        normalized_reason = reason if reason.endswith(".") else f"{reason}."
        parts.append(normalized_reason)
    if remove_after:
        parts.append(f"remove_after={remove_after}.")
    return " ".join(parts)


def resolve_model_alias(name: str) -> tuple[str, str | None]:
    requested_name = str(name).strip()
    alias_meta = _DEPRECATED_MODEL_ALIASES.get(requested_name)
    if not isinstance(alias_meta, Mapping):
        return requested_name, None
    resolved_name = str(alias_meta.get("target", requested_name)).strip() or requested_name
    warning_message = _format_alias_warning(
        requested_name=requested_name,
        resolved_name=resolved_name,
        alias_meta={str(k): str(v) for k, v in alias_meta.items()},
    )
    return resolved_name, warning_message


@dataclass
class _PredictVnAdapter:
    _model: Any

    def __getattr__(self, item: str) -> Any:
        return getattr(self._model, item)

    def predict_vn(
        self,
        phi: Any,  # noqa: ARG002
        conditions: ConditionMap,
        step_index: int = 0,  # noqa: ARG002
    ) -> float:
        return float(self._model.predict(conditions))


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    return MODEL_REGISTRY.register(name)


def list_models() -> list[str]:
    return MODEL_REGISTRY.list()


def make_model(name: str, **kwargs: object) -> Model:
    resolved_name, alias_warning = resolve_model_alias(str(name))
    if alias_warning is not None:
        warnings.warn(alias_warning, FutureWarning, stacklevel=2)
    model = MODEL_REGISTRY.create(resolved_name, **kwargs)
    if not hasattr(model, "predict"):
        raise TypeError(f"model: '{resolved_name}' does not implement predict(...)")
    if not hasattr(model, "predict_vn"):
        model = _PredictVnAdapter(model)
    return model  # type: ignore[return-value]


@dataclass
class ConstantVnModel:
    """Baseline Vn predictor with optional linear condition term."""

    default_value: float = 0.0
    mean_value: float | None = None
    condition_weights: Mapping[str, float] | None = None

    def fit(self, features: Sequence[FeatureMap], targets: Sequence[float]) -> None:
        del features  # baseline intentionally ignores feature vectors.
        if not targets:
            self.mean_value = self.default_value
            return
        self.mean_value = sum(float(value) for value in targets) / len(targets)

    def _condition_term(self, features: FeatureMap) -> float:
        if not self.condition_weights:
            return 0.0

        total = 0.0
        for key, weight in self.condition_weights.items():
            total += float(weight) * float(features.get(key, 0.0))
        return total

    def predict(self, features: FeatureMap) -> float:
        base = self.default_value if self.mean_value is None else self.mean_value
        return float(base) + self._condition_term(features)

    def predict_vn(
        self,
        phi: Any,  # noqa: ARG002 - reserved for future spatially varying models.
        conditions: ConditionMap,
        step_index: int = 0,  # noqa: ARG002 - reserved for time-dependent models.
    ) -> float:
        return self.predict(conditions)


@dataclass
class TrainableLinearVnModel:
    """Trainable linear regressor that stays dependency-free."""

    default_value: float = 0.0
    learning_rate: float = 0.05
    epochs: int = 120
    l2: float = 1e-6
    max_grad_norm: float = 10.0
    bias: float | None = None
    feature_weights: dict[str, float] | None = None
    feature_scales: dict[str, float] | None = None

    def fit(self, features: Sequence[FeatureMap], targets: Sequence[float]) -> None:
        if len(features) != len(targets):
            raise ValueError("features and targets length must match")
        if not targets:
            self.bias = self.default_value
            self.feature_weights = {}
            self.feature_scales = {}
            return

        keys = sorted({str(key) for sample in features for key in sample.keys()})
        weights: dict[str, float] = {key: 0.0 for key in keys}
        scales: dict[str, float] = {}
        for key in keys:
            max_abs = max(abs(float(sample.get(key, 0.0))) for sample in features)
            scales[key] = max(1.0, max_abs)
        bias = float(self.default_value)
        lr = max(float(self.learning_rate), 1e-8)
        grad_cap = max(float(self.max_grad_norm), 1e-6)
        n = float(len(targets))
        for _ in range(max(int(self.epochs), 1)):
            grad_bias = 0.0
            grad_w = {key: 0.0 for key in keys}
            for sample, target in zip(features, targets):
                pred = bias
                for key in keys:
                    pred += weights[key] * (float(sample.get(key, 0.0)) / scales[key])
                err = pred - float(target)
                if not math.isfinite(err):
                    raise ValueError("non-finite error encountered during linear fit")
                grad_bias += 2.0 * err
                for key in keys:
                    grad_w[key] += 2.0 * err * (float(sample.get(key, 0.0)) / scales[key])
            grad_bias = max(-grad_cap, min(grad_cap, grad_bias / n))
            bias -= lr * grad_bias
            for key in keys:
                grad = (grad_w[key] / n) + 2.0 * float(self.l2) * weights[key]
                grad = max(-grad_cap, min(grad_cap, grad))
                weights[key] -= lr * grad
                if not math.isfinite(weights[key]):
                    raise ValueError("non-finite weight encountered during linear fit")

        self.bias = bias
        self.feature_weights = weights
        self.feature_scales = scales

    def predict(self, features: FeatureMap) -> float:
        pred = float(self.default_value if self.bias is None else self.bias)
        scales = self.feature_scales or {}
        for key, weight in (self.feature_weights or {}).items():
            scale = max(1.0, float(scales.get(key, 1.0)))
            pred += float(weight) * (float(features.get(key, 0.0)) / scale)
        return pred

    def predict_vn(
        self,
        phi: Any,  # noqa: ARG002 - reserved for future spatially varying models.
        conditions: ConditionMap,
        step_index: int = 0,  # noqa: ARG002 - reserved for time-dependent models.
    ) -> float:
        return self.predict(conditions)


def _drop_privileged_channels(features: FeatureMap) -> dict[str, float]:
    projected: dict[str, float] = {}
    for key, value in features.items():
        name = str(key)
        if name.startswith("priv_"):
            continue
        projected[name] = float(value)
    return projected


@dataclass
class SparseVnStudentModel:
    """Legacy tabular student wrapper kept for backward-compatible aliases."""

    default_value: float = 0.0
    learning_rate: float = 0.02
    epochs: int = 140
    l2: float = 1e-6
    max_grad_norm: float = 10.0
    _model: TrainableLinearVnModel | None = None

    def fit(self, features: Sequence[FeatureMap], targets: Sequence[float]) -> None:
        model = TrainableLinearVnModel(
            default_value=self.default_value,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            l2=self.l2,
            max_grad_norm=self.max_grad_norm,
        )
        projected = [_drop_privileged_channels(sample) for sample in features]
        model.fit(projected, targets)
        self._model = model

    def predict(self, features: FeatureMap) -> float:
        if self._model is None:
            return float(self.default_value)
        return float(self._model.predict(_drop_privileged_channels(features)))

    def predict_vn(
        self,
        phi: Any,  # noqa: ARG002
        conditions: ConditionMap,
        step_index: int = 0,  # noqa: ARG002
    ) -> float:
        return self.predict(conditions)


@dataclass
class SparseVnTeacherModel:
    """Legacy tabular teacher wrapper kept for backward-compatible aliases."""

    default_value: float = 0.0
    learning_rate: float = 0.02
    epochs: int = 160
    l2: float = 1e-6
    max_grad_norm: float = 10.0
    _model: TrainableLinearVnModel | None = None

    def fit(self, features: Sequence[FeatureMap], targets: Sequence[float]) -> None:
        model = TrainableLinearVnModel(
            default_value=self.default_value,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            l2=self.l2,
            max_grad_norm=self.max_grad_norm,
        )
        model.fit(features, targets)
        self._model = model

    def predict(self, features: FeatureMap) -> float:
        if self._model is None:
            return float(self.default_value)
        return float(self._model.predict(features))

    def predict_vn(
        self,
        phi: Any,  # noqa: ARG002
        conditions: ConditionMap,
        step_index: int = 0,  # noqa: ARG002
    ) -> float:
        return self.predict(conditions)


@dataclass
class SurfaceGraphVnModel:
    """Surface-point graph model with non-local coupling and SDF projection."""

    default_value: float = 0.0
    condition_weights: Mapping[str, float] | None = None
    surface_band_width: float = 0.75
    projection_band_width: float = 2.0
    local_radius: float = 2.5
    nonlocal_radius: float = 9.0
    max_nonlocal_degree: int = 8
    local_strength: float = 0.02
    nonlocal_strength: float = 0.25
    reflection_decay: float = 4.0
    min_grad_norm: float = 1e-6
    mean_value: float | None = None

    def fit(self, features: Sequence[FeatureMap], targets: Sequence[float]) -> None:
        del features  # not used by this geometry-driven baseline
        if not targets:
            self.mean_value = self.default_value
            return
        self.mean_value = sum(float(value) for value in targets) / len(targets)

    def _condition_term(self, features: FeatureMap) -> float:
        if not self.condition_weights:
            return 0.0
        total = 0.0
        for key, weight in self.condition_weights.items():
            total += float(weight) * float(features.get(key, 0.0))
        return total

    def predict(self, features: FeatureMap) -> float:
        base = self.default_value if self.mean_value is None else self.mean_value
        return float(base) + self._condition_term(features)

    def _surface_nodes(self, phi_arr: Any) -> tuple[Any, Any]:
        np = _np
        if np is None:  # pragma: no cover - numpy-free fallback
            raise RuntimeError("numpy is required for surface graph computation")
        grad = finite_diff_grad(phi_arr)
        grad_norm = np.linalg.norm(grad, axis=0)
        mask = (np.abs(phi_arr) <= float(self.surface_band_width)) & (
            grad_norm >= max(float(self.min_grad_norm), 1e-12)
        )
        coords = np.argwhere(mask)
        if coords.size == 0:
            return coords, coords

        grad_last = np.moveaxis(grad, 0, -1)
        grad_samples = grad_last[tuple(coords.T)]
        norm_samples = grad_norm[tuple(coords.T)]
        normals = grad_samples / np.maximum(norm_samples[:, None], 1e-12)
        return coords.astype(float), normals

    def _graph_signals(self, coords: Any, normals: Any) -> tuple[Any, Any]:
        np = _np
        if np is None:  # pragma: no cover - numpy-free fallback
            raise RuntimeError("numpy is required for surface graph computation")
        count = int(coords.shape[0])
        if count == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

        vectors = coords[None, :, :] - coords[:, None, :]
        dist = np.linalg.norm(vectors, axis=-1)
        safe_dist = np.maximum(dist, 1e-12)
        unit_dirs = vectors / safe_dist[:, :, None]

        local_mask = (dist > 0.0) & (dist <= max(float(self.local_radius), 1e-6))
        local_facing = np.maximum(0.0, np.sum(normals[:, None, :] * unit_dirs, axis=-1))
        local_count = np.sum(local_mask, axis=1)
        local_signal = np.sum(local_facing * local_mask, axis=1) / np.maximum(local_count, 1.0)

        candidate = (
            (dist > max(float(self.local_radius), 1e-6))
            & (dist <= max(float(self.nonlocal_radius), float(self.local_radius)))
        )
        facing_i = np.maximum(0.0, -np.sum(normals[:, None, :] * unit_dirs, axis=-1))
        facing_j = np.maximum(0.0, np.sum(normals[None, :, :] * unit_dirs, axis=-1))
        visibility = facing_i * facing_j
        reflect = visibility * np.exp(-dist / max(float(self.reflection_decay), 1e-6))
        reflect *= candidate
        np.fill_diagonal(reflect, 0.0)

        if int(self.max_nonlocal_degree) > 0:
            pruned = np.zeros_like(reflect)
            topk = min(int(self.max_nonlocal_degree), count - 1)
            if topk > 0:
                for idx in range(count):
                    row = reflect[idx]
                    if not np.any(row > 0.0):
                        continue
                    selected = np.argpartition(row, -topk)[-topk:]
                    pruned[idx, selected] = row[selected]
            reflect = pruned
        reflect_norm = np.sum(reflect, axis=1)
        nonlocal_signal = reflect_norm / np.maximum(np.sum(reflect > 0.0, axis=1), 1.0)
        return local_signal, nonlocal_signal

    def _project_to_grid(self, phi_arr: Any, coords: Any, surface_vn: Any, base_vn: float) -> Any:
        np = _np
        if np is None:  # pragma: no cover - numpy-free fallback
            raise RuntimeError("numpy is required for surface graph computation")
        vn_flat = np.full(phi_arr.size, float(base_vn), dtype=float)
        if coords.size == 0:
            return vn_flat.reshape(phi_arr.shape)

        all_coords = np.argwhere(np.ones_like(phi_arr, dtype=bool))
        band = max(float(self.projection_band_width), 0.0)
        if band > 0.0:
            active_mask = np.abs(phi_arr.ravel()) <= band
        else:
            active_mask = np.ones(phi_arr.size, dtype=bool)
        active_coords = all_coords[active_mask]
        if active_coords.size == 0:
            return vn_flat.reshape(phi_arr.shape)

        chunk = 2048
        cursor = 0
        while cursor < active_coords.shape[0]:
            stop = min(cursor + chunk, active_coords.shape[0])
            points = active_coords[cursor:stop].astype(float)
            diff = points[:, None, :] - coords[None, :, :]
            dist2 = np.sum(diff * diff, axis=-1)
            nearest = np.argmin(dist2, axis=1)
            vn_flat[np.flatnonzero(active_mask)[cursor:stop]] = surface_vn[nearest]
            cursor = stop
        return vn_flat.reshape(phi_arr.shape)

    def predict_vn(
        self,
        phi: Any,
        conditions: ConditionMap,
        step_index: int = 0,  # noqa: ARG002 - reserved for future time-conditioning.
    ) -> Any:
        base_vn = self.predict(conditions)
        np = _np
        if np is None:  # pragma: no cover - graceful fallback without numpy
            return base_vn

        phi_arr = np.asarray(phi, dtype=float)
        if phi_arr.ndim < 2:
            raise ValueError("surface_graph_vn expects phi with at least 2 dimensions")

        coords, normals = self._surface_nodes(phi_arr)
        if coords.size == 0:
            return np.full_like(phi_arr, float(base_vn), dtype=float)
        local_signal, nonlocal_signal = self._graph_signals(coords, normals)
        surface_vn = (
            float(base_vn)
            + float(self.local_strength) * local_signal
            + float(self.nonlocal_strength) * nonlocal_signal
        )
        return self._project_to_grid(phi_arr, coords, surface_vn, float(base_vn))


@dataclass
class TimeConditionedOperatorModel:
    """Direct operator baseline for phi(t) conditioned on (phi0, c, t)."""

    default_vn: float = 0.0
    l2: float = 1e-6
    mean_vn: float | None = None
    operator_weights: Any | None = None
    phi_shape: tuple[int, ...] | None = None
    condition_keys: tuple[str, ...] | None = None

    def fit(self, features: Sequence[FeatureMap], targets: Sequence[float]) -> None:
        del features
        if not targets:
            self.mean_vn = self.default_vn
            return
        self.mean_vn = sum(float(value) for value in targets) / len(targets)

    def predict(self, features: FeatureMap) -> float:
        del features
        return float(self.default_vn if self.mean_vn is None else self.mean_vn)

    def _feature_vector(
        self,
        phi0: Any,
        conditions: Mapping[str, float],
        t: float,
        condition_keys: Sequence[str],
    ) -> Any:
        np = _require_numpy()
        phi_flat = np.asarray(phi0, dtype=float).reshape(-1)
        condition_vec = np.asarray(
            [float(conditions.get(key, 0.0)) for key in condition_keys],
            dtype=float,
        )
        time_and_bias = np.asarray([float(t), 1.0], dtype=float)
        return np.concatenate([phi_flat, condition_vec, time_and_bias], axis=0)

    def fit_operator(self, runs: Sequence[Any]) -> None:
        np = _require_numpy()
        if not runs:
            raise ValueError("operator fit requires at least one run")

        first_phi0 = np.asarray(runs[0].phi_t[0], dtype=float)
        if first_phi0.ndim < 2:
            raise ValueError("operator baseline expects 2D or 3D SDF grids")
        phi_shape = tuple(int(dim) for dim in first_phi0.shape)
        condition_keys = tuple(
            sorted({str(key) for run in runs for key in run.recipe.keys()})
        )

        samples: list[Any] = []
        targets: list[Any] = []
        vn_targets: list[float] = []
        for run in runs:
            if not run.phi_t:
                raise ValueError("run.phi_t must have at least one frame")
            dt = float(run.dt)
            if dt <= 0.0:
                raise ValueError("run.dt must be positive")

            phi0 = np.asarray(run.phi_t[0], dtype=float)
            if tuple(phi0.shape) != phi_shape:
                raise ValueError("all runs must share the same phi grid shape")
            recipe = {str(key): float(value) for key, value in run.recipe.items()}

            for step_index, frame in enumerate(run.phi_t):
                frame_arr = np.asarray(frame, dtype=float)
                if tuple(frame_arr.shape) != phi_shape:
                    raise ValueError("all frames must share the same phi grid shape")
                samples.append(
                    self._feature_vector(
                        phi0=phi0,
                        conditions=recipe,
                        t=float(step_index) * dt,
                        condition_keys=condition_keys,
                    )
                )
                targets.append(frame_arr.reshape(-1))
                if step_index + 1 < len(run.phi_t):
                    next_arr = np.asarray(run.phi_t[step_index + 1], dtype=float)
                    vn_targets.append(
                        float((np.mean(frame_arr) - np.mean(next_arr)) / dt)
                    )

        x_mat = np.stack(samples, axis=0)
        y_mat = np.stack(targets, axis=0)
        reg = np.eye(x_mat.shape[1], dtype=float) * max(float(self.l2), 0.0)
        reg[-1, -1] = 0.0  # Keep intercept weakly constrained.
        self.operator_weights = np.linalg.pinv(x_mat.T @ x_mat + reg) @ x_mat.T @ y_mat
        self.phi_shape = phi_shape
        self.condition_keys = condition_keys
        self.mean_vn = (
            float(sum(vn_targets) / len(vn_targets))
            if vn_targets
            else float(self.default_vn)
        )

    def predict_phi(
        self,
        phi0: Any,
        conditions: Mapping[str, float] | None,
        t: float,
    ) -> Any:
        np = _require_numpy()
        if self.operator_weights is None or self.phi_shape is None:
            raise RuntimeError("operator model is not trained; call fit_operator(...) first")

        phi0_arr = np.asarray(phi0, dtype=float)
        if tuple(phi0_arr.shape) != self.phi_shape:
            raise ValueError(
                f"phi0 shape mismatch: expected {self.phi_shape}, got {tuple(phi0_arr.shape)}"
            )
        cond = {} if conditions is None else {str(k): float(v) for k, v in conditions.items()}
        keys = self.condition_keys or ()
        feature = self._feature_vector(
            phi0=phi0_arr,
            conditions=cond,
            t=float(t),
            condition_keys=keys,
        )
        prediction = feature @ self.operator_weights
        return prediction.reshape(self.phi_shape)

    def predict_vn(
        self,
        phi: Any,  # noqa: ARG002 - direct operator predicts phi(t) instead.
        conditions: ConditionMap,  # noqa: ARG002 - compatibility path.
        step_index: int = 0,  # noqa: ARG002 - compatibility path.
    ) -> float:
        return self.predict({})


@register_model("baseline_vn_constant")
def _build_baseline_vn_constant_model(default_value: float = 0.0) -> Model:
    return ConstantVnModel(default_value=default_value)


@register_model("baseline_vn_linear")
def _build_baseline_vn_linear_model(
    default_value: float = 0.0,
    condition_weights: Mapping[str, float] | None = None,
) -> Model:
    return ConstantVnModel(
        default_value=default_value,
        condition_weights=condition_weights,
    )


@register_model("baseline_mean")
def _build_baseline_mean_model(default_value: float = 0.0) -> Model:
    # Compatibility alias used by earlier checks/tasks.
    return ConstantVnModel(default_value=default_value)


@register_model("baseline_vn_linear_trainable")
def _build_baseline_vn_linear_trainable_model(
    default_value: float = 0.0,
    learning_rate: float = 0.05,
    epochs: int = 120,
    l2: float = 1e-6,
) -> Model:
    return TrainableLinearVnModel(
        default_value=default_value,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )


@register_model("surface_graph_vn")
def _build_surface_graph_vn_model(
    default_value: float = 0.0,
    condition_weights: Mapping[str, float] | None = None,
    surface_band_width: float = 0.75,
    projection_band_width: float = 2.0,
    local_radius: float = 2.5,
    nonlocal_radius: float = 9.0,
    max_nonlocal_degree: int = 8,
    local_strength: float = 0.02,
    nonlocal_strength: float = 0.25,
    reflection_decay: float = 4.0,
) -> Model:
    return SurfaceGraphVnModel(
        default_value=default_value,
        condition_weights=condition_weights,
        surface_band_width=surface_band_width,
        projection_band_width=projection_band_width,
        local_radius=local_radius,
        nonlocal_radius=nonlocal_radius,
        max_nonlocal_degree=max_nonlocal_degree,
        local_strength=local_strength,
        nonlocal_strength=nonlocal_strength,
        reflection_decay=reflection_decay,
    )


@register_model("operator_time_conditioned")
def _build_operator_time_conditioned_model(
    default_value: float = 0.0,
    l2: float = 1e-6,
) -> Model:
    return TimeConditionedOperatorModel(default_vn=default_value, l2=l2)


@register_model("sparse_vn_student_linear_legacy")
def _build_sparse_vn_student_linear_legacy_model(
    default_value: float = 0.0,
    learning_rate: float = 0.02,
    epochs: int = 140,
    l2: float = 1e-6,
) -> Model:
    return SparseVnStudentModel(
        default_value=default_value,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )


@register_model("sparse_vn_student")
def _build_sparse_vn_student_model(
    default_value: float = 0.0,
    learning_rate: float = 0.02,
    epochs: int = 140,
    l2: float = 1e-6,
) -> Model:
    return _build_sparse_vn_student_linear_legacy_model(
        default_value=default_value,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )


@register_model("sparse_vn_teacher_linear_legacy")
def _build_sparse_vn_teacher_linear_legacy_model(
    default_value: float = 0.0,
    learning_rate: float = 0.02,
    epochs: int = 160,
    l2: float = 1e-6,
) -> Model:
    return SparseVnTeacherModel(
        default_value=default_value,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )


@register_model("sparse_vn_teacher")
def _build_sparse_vn_teacher_model(
    default_value: float = 0.0,
    learning_rate: float = 0.02,
    epochs: int = 160,
    l2: float = 1e-6,
) -> Model:
    return _build_sparse_vn_teacher_linear_legacy_model(
        default_value=default_value,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
