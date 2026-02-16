from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Protocol

from wafer_surrogate.registries import Registry


NumericMap = dict[str, float]


class PreprocessStep(Protocol):
    def transform(self, payload: NumericMap) -> NumericMap:
        ...


PreprocessFactory = Callable[..., PreprocessStep]
PREPROCESS_STEP_REGISTRY = Registry[PreprocessFactory]("preprocess_step")


def register_preprocess_step(name: str) -> Callable[[PreprocessFactory], PreprocessFactory]:
    return PREPROCESS_STEP_REGISTRY.register(name)


def list_preprocess_steps() -> list[str]:
    return PREPROCESS_STEP_REGISTRY.list()


def make_preprocess_step(name: str, **kwargs: object) -> PreprocessStep:
    step = PREPROCESS_STEP_REGISTRY.create(name, **kwargs)
    if not hasattr(step, "transform"):
        raise TypeError(f"preprocess_step: '{name}' does not implement transform(...)")
    return step  # type: ignore[return-value]


StepSpec = str | tuple[str, Mapping[str, object]]


class PreprocessPipeline:
    """Composable pipeline; applies steps in the order passed by caller."""

    def __init__(self, steps: Sequence[PreprocessStep]) -> None:
        self._steps = list(steps)

    def transform(self, payload: Mapping[str, float]) -> NumericMap:
        output: NumericMap = {key: float(value) for key, value in payload.items()}
        for step in self._steps:
            output = step.transform(output)
        return output


def build_preprocess_pipeline(step_specs: Sequence[StepSpec]) -> PreprocessPipeline:
    steps: list[PreprocessStep] = []
    for spec in step_specs:
        if isinstance(spec, str):
            step_name = spec
            params: Mapping[str, object] = {}
        else:
            step_name, params = spec
        steps.append(make_preprocess_step(step_name, **dict(params)))
    return PreprocessPipeline(steps)


@dataclass(frozen=True)
class FeatureScaler:
    means: dict[str, float]
    stds: dict[str, float]

    def transform_row(self, row: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in self.means:
            value = float(row.get(key, 0.0))
            mean = float(self.means[key])
            std = float(self.stds.get(key, 1.0))
            scale = std if abs(std) > 1e-12 else 1.0
            out[key] = (value - mean) / scale
        for key, value in row.items():
            if key not in out:
                out[str(key)] = float(value)
        return out

    def inverse_row(self, row: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in self.means:
            value = float(row.get(key, 0.0))
            mean = float(self.means[key])
            std = float(self.stds.get(key, 1.0))
            scale = std if abs(std) > 1e-12 else 1.0
            out[key] = (value * scale) + mean
        for key, value in row.items():
            if key not in out:
                out[str(key)] = float(value)
        return out

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "mode": "standard",
            "means": dict(self.means),
            "stds": dict(self.stds),
        }


@dataclass(frozen=True)
class RobustFeatureScaler:
    medians: dict[str, float]
    iqrs: dict[str, float]

    def transform_row(self, row: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in self.medians:
            value = float(row.get(key, 0.0))
            median = float(self.medians[key])
            iqr = float(self.iqrs.get(key, 1.0))
            scale = iqr if abs(iqr) > 1e-12 else 1.0
            out[key] = (value - median) / scale
        for key, value in row.items():
            if key not in out:
                out[str(key)] = float(value)
        return out

    def inverse_row(self, row: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in self.medians:
            value = float(row.get(key, 0.0))
            median = float(self.medians[key])
            iqr = float(self.iqrs.get(key, 1.0))
            scale = iqr if abs(iqr) > 1e-12 else 1.0
            out[key] = (value * scale) + median
        for key, value in row.items():
            if key not in out:
                out[str(key)] = float(value)
        return out

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "mode": "robust",
            "medians": dict(self.medians),
            "iqrs": dict(self.iqrs),
        }


@dataclass(frozen=True)
class QuantileFeatureTransformer:
    sorted_values: dict[str, list[float]]

    def transform_row(self, row: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, values in self.sorted_values.items():
            if not values:
                out[key] = 0.0
                continue
            x = float(row.get(key, 0.0))
            idx = bisect.bisect_left(values, x)
            if len(values) <= 1:
                out[key] = 0.5
            else:
                out[key] = float(idx) / float(len(values) - 1)
        for key, value in row.items():
            if key not in out:
                out[str(key)] = float(value)
        return out

    def inverse_row(self, row: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, values in self.sorted_values.items():
            if not values:
                out[key] = 0.0
                continue
            q = max(0.0, min(1.0, float(row.get(key, 0.0))))
            if len(values) == 1:
                out[key] = float(values[0])
                continue
            pos = q * float(len(values) - 1)
            lo = int(pos)
            hi = min(lo + 1, len(values) - 1)
            frac = pos - float(lo)
            out[key] = (1.0 - frac) * float(values[lo]) + frac * float(values[hi])
        for key, value in row.items():
            if key not in out:
                out[str(key)] = float(value)
        return out

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": "quantile",
            "sorted_values": {key: list(values) for key, values in self.sorted_values.items()},
        }


@dataclass(frozen=True)
class PCAProjector:
    keys: list[str]
    means: list[float]
    components: list[list[float]]

    def transform_row(self, row: Mapping[str, float]) -> dict[str, float]:
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("numpy is required for PCA preprocessing") from exc
        vec = np.asarray([float(row.get(key, 0.0)) for key in self.keys], dtype=float)
        center = vec - np.asarray(self.means, dtype=float)
        comp = np.asarray(self.components, dtype=float)
        proj = comp @ center
        out: dict[str, float] = {f"pca_{idx}": float(value) for idx, value in enumerate(proj.tolist())}
        return out

    def inverse_row(self, row: Mapping[str, float]) -> dict[str, float]:
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("numpy is required for PCA preprocessing") from exc
        proj = np.asarray([float(row.get(f"pca_{idx}", 0.0)) for idx in range(len(self.components))], dtype=float)
        comp = np.asarray(self.components, dtype=float)
        vec = np.asarray(self.means, dtype=float) + (comp.T @ proj)
        return {key: float(vec[idx]) for idx, key in enumerate(self.keys)}

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": "pca",
            "keys": list(self.keys),
            "means": list(self.means),
            "components": [list(row) for row in self.components],
        }


@dataclass(frozen=True)
class TargetTransformer:
    mode: str
    mean: float = 0.0
    std: float = 1.0
    shift: float = 0.0

    def transform(self, values: Sequence[float]) -> list[float]:
        if self.mode == "identity":
            return [float(value) for value in values]
        if self.mode == "log1p":
            return [math.log1p(float(value) + float(self.shift)) for value in values]
        scale = self.std if abs(self.std) > 1e-12 else 1.0
        return [(float(value) - self.mean) / scale for value in values]

    def inverse(self, values: Sequence[float]) -> list[float]:
        if self.mode == "identity":
            return [float(value) for value in values]
        if self.mode == "log1p":
            return [math.expm1(float(value)) - float(self.shift) for value in values]
        scale = self.std if abs(self.std) > 1e-12 else 1.0
        return [(float(value) * scale) + self.mean for value in values]

    def to_dict(self) -> dict[str, float | str]:
        return {
            "mode": self.mode,
            "mean": float(self.mean),
            "std": float(self.std),
            "shift": float(self.shift),
        }


def fit_feature_scaler(rows: Sequence[Mapping[str, float]]) -> FeatureScaler:
    if not rows:
        return FeatureScaler(means={}, stds={})
    keys = sorted({str(key) for row in rows for key in row.keys()})
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        means[key] = fmean(values)
        stds[key] = pstdev(values) if len(values) > 1 else 1.0
        if abs(stds[key]) <= 1e-12:
            stds[key] = 1.0
    return FeatureScaler(means=means, stds=stds)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    qq = max(0.0, min(1.0, float(q)))
    pos = qq * float(len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - float(lo)
    return (1.0 - frac) * float(sorted_values[lo]) + frac * float(sorted_values[hi])


def fit_robust_feature_scaler(rows: Sequence[Mapping[str, float]]) -> RobustFeatureScaler:
    if not rows:
        return RobustFeatureScaler(medians={}, iqrs={})
    keys = sorted({str(key) for row in rows for key in row.keys()})
    medians: dict[str, float] = {}
    iqrs: dict[str, float] = {}
    for key in keys:
        values = sorted(float(row.get(key, 0.0)) for row in rows)
        q1 = _percentile(values, 0.25)
        q2 = _percentile(values, 0.50)
        q3 = _percentile(values, 0.75)
        iqr = q3 - q1
        medians[key] = float(q2)
        iqrs[key] = float(iqr) if abs(float(iqr)) > 1e-12 else 1.0
    return RobustFeatureScaler(medians=medians, iqrs=iqrs)


def fit_quantile_feature_transformer(rows: Sequence[Mapping[str, float]]) -> QuantileFeatureTransformer:
    if not rows:
        return QuantileFeatureTransformer(sorted_values={})
    keys = sorted({str(key) for row in rows for key in row.keys()})
    values_by_key: dict[str, list[float]] = {}
    for key in keys:
        values_by_key[key] = sorted(float(row.get(key, 0.0)) for row in rows)
    return QuantileFeatureTransformer(sorted_values=values_by_key)


def fit_pca_projector(rows: Sequence[Mapping[str, float]], n_components: int = 3) -> PCAProjector:
    if not rows:
        return PCAProjector(keys=[], means=[], components=[])
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for PCA preprocessing") from exc

    keys = sorted({str(key) for row in rows for key in row.keys()})
    if not keys:
        return PCAProjector(keys=[], means=[], components=[])
    x = np.asarray([[float(row.get(key, 0.0)) for key in keys] for row in rows], dtype=float)
    means = np.mean(x, axis=0)
    centered = x - means
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = max(1, min(int(n_components), int(vt.shape[0]), int(vt.shape[1])))
    components = vt[:k, :]
    return PCAProjector(
        keys=list(keys),
        means=[float(v) for v in means.tolist()],
        components=[[float(v) for v in row] for row in components.tolist()],
    )


def fit_target_transform(values: Sequence[float], mode: str = "identity") -> TargetTransformer:
    mode_name = str(mode).strip().lower() or "identity"
    if mode_name not in {"identity", "standard", "log1p"}:
        raise ValueError(f"unsupported target transform mode: {mode_name}")
    if mode_name == "identity" or not values:
        return TargetTransformer(mode="identity", mean=0.0, std=1.0, shift=0.0)
    if mode_name == "log1p":
        vals = [float(value) for value in values]
        min_value = min(vals)
        shift = (-min_value + 1e-6) if min_value <= -1.0 else 0.0
        return TargetTransformer(mode="log1p", mean=0.0, std=1.0, shift=shift)
    vals = [float(value) for value in values]
    mean = fmean(vals)
    std = pstdev(vals) if len(vals) > 1 else 1.0
    if abs(std) <= 1e-12:
        std = 1.0
    return TargetTransformer(mode="standard", mean=mean, std=std, shift=0.0)


@dataclass
class IdentityStep:
    def transform(self, payload: NumericMap) -> NumericMap:
        return dict(payload)


@dataclass
class ScaleStep:
    factor: float = 1.0
    field: str | None = None

    def transform(self, payload: NumericMap) -> NumericMap:
        output = dict(payload)
        if self.field is None:
            for key, value in output.items():
                output[key] = float(value) * self.factor
            return output

        if self.field in output:
            output[self.field] = float(output[self.field]) * self.factor
        return output


@dataclass
class OffsetStep:
    value: float = 0.0
    field: str | None = None

    def transform(self, payload: NumericMap) -> NumericMap:
        output = dict(payload)
        if self.field is None:
            for key, current in output.items():
                output[key] = float(current) + self.value
            return output

        if self.field in output:
            output[self.field] = float(output[self.field]) + self.value
        return output


@register_preprocess_step("identity")
def _build_identity_step() -> PreprocessStep:
    return IdentityStep()


@register_preprocess_step("scale")
def _build_scale_step(factor: float = 1.0, field: str | None = None) -> PreprocessStep:
    return ScaleStep(factor=factor, field=field)


@register_preprocess_step("offset")
def _build_offset_step(value: float = 0.0, field: str | None = None) -> PreprocessStep:
    return OffsetStep(value=value, field=field)
