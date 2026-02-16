from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from wafer_surrogate.registries import Registry


ShapeState = Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]]


class ObservationModel(Protocol):
    def project(self, shape_state: ShapeState) -> list[float]:
        ...

    def feature_names(self) -> list[str]:
        ...


class ObservationModelError(ValueError):
    """Invalid inputs/outputs for observation feature projection."""


ObservationFactory = Callable[..., ObservationModel]
OBSERVATION_MODEL_REGISTRY = Registry[ObservationFactory]("observation_model")


def register_observation_model(name: str) -> Callable[[ObservationFactory], ObservationFactory]:
    return OBSERVATION_MODEL_REGISTRY.register(name)


def list_observation_models() -> list[str]:
    return OBSERVATION_MODEL_REGISTRY.list()


def make_observation_model(name: str, **kwargs: object) -> ObservationModel:
    model = OBSERVATION_MODEL_REGISTRY.create(name, **kwargs)
    if not hasattr(model, "project"):
        raise TypeError(f"observation_model: '{name}' must implement project(shape_state)")
    return model  # type: ignore[return-value]
