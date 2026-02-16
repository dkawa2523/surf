from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Minimal plugin registry with decorator-based registration."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: dict[str, T] = {}

    def register(self, key: str | None = None) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            item_key = key or getattr(item, "__name__", None)
            if not item_key:
                raise ValueError(f"{self._name}: registry key is required")
            self.add(item_key, item)
            return item

        return decorator

    def add(self, key: str, item: T) -> None:
        if key in self._items:
            raise ValueError(f"{self._name}: '{key}' is already registered")
        self._items[key] = item

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(self.list()) or "(empty)"
            raise KeyError(
                f"{self._name}: '{key}' is not registered. available={available}"
            ) from exc

    def create(self, key: str, *args: object, **kwargs: object) -> object:
        factory = self.get(key)
        if not callable(factory):
            raise TypeError(f"{self._name}: registered item '{key}' is not callable")
        return factory(*args, **kwargs)

    def list(self) -> list[str]:
        return sorted(self._items.keys())

    def __len__(self) -> int:
        return len(self._items)

