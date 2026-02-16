from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


class SemFeatureError(ValueError):
    """Invalid SEM feature payload."""


@dataclass(frozen=True)
class SemFeatureVector:
    y: list[float]
    feature_names: list[str] | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.y:
            raise SemFeatureError("y must contain at least one value")
        if any(not math.isfinite(float(v)) for v in self.y):
            raise SemFeatureError("y must contain finite values")
        if self.feature_names is None:
            return
        if len(self.feature_names) != len(self.y):
            raise SemFeatureError("feature_names length must match y length")
        if any(not str(name).strip() for name in self.feature_names):
            raise SemFeatureError("feature_names must be non-empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise SemFeatureError("feature_names must be unique")


def _float_list(value: object, *, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SemFeatureError(f"{name} must be a sequence")
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise SemFeatureError(f"{name} must contain numeric values") from exc
    if not out:
        raise SemFeatureError(f"{name} must contain at least one value")
    if any(not math.isfinite(v) for v in out):
        raise SemFeatureError(f"{name} must contain finite values")
    return out


def _from_mapping(payload: Mapping[str, object]) -> SemFeatureVector:
    if "y" in payload:
        y = _float_list(payload["y"], name="y")
        names = payload.get("feature_names")
        return SemFeatureVector(
            y=y,
            feature_names=(
                [str(n) for n in names]
                if isinstance(names, Sequence) and not isinstance(names, (str, bytes, bytearray))
                else None
            ),
        )
    data = payload.get("features", payload)
    if not isinstance(data, Mapping) or not data:
        raise SemFeatureError("mapping payload must contain feature values")
    names = [str(k) for k in data.keys()]
    return SemFeatureVector(y=[float(v) for v in data.values()], feature_names=names)


def _from_payload(payload: object) -> SemFeatureVector:
    if isinstance(payload, Mapping):
        return _from_mapping(payload)
    return SemFeatureVector(y=_float_list(payload, name="y"))


def read_sem_features_json(path: str | Path) -> SemFeatureVector:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fp:
        vec = _from_payload(json.load(fp))
    return SemFeatureVector(y=vec.y, feature_names=vec.feature_names, source=str(p))


def _try_float_row(row: Sequence[str]) -> list[float] | None:
    try:
        return [float(c) for c in row]
    except ValueError:
        return None


def read_sem_features_csv(path: str | Path) -> SemFeatureVector:
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as fp:
        rows = [[c.strip() for c in row] for row in csv.reader(fp) if any(c.strip() for c in row)]
    if not rows:
        raise SemFeatureError("csv must contain at least one non-empty row")
    if len({len(row) for row in rows}) != 1:
        raise SemFeatureError("csv rows must share the same column count")

    if len(rows) == 2 and _try_float_row(rows[1]) is not None and _try_float_row(rows[0]) is None:
        return SemFeatureVector(y=_float_list(_try_float_row(rows[1]), name="y"), feature_names=rows[0], source=str(p))

    if all(len(row) == 2 for row in rows):
        second_col = _try_float_row([row[1] for row in rows])
        if second_col is not None:
            return SemFeatureVector(y=_float_list(second_col, name="y"), feature_names=[row[0] for row in rows], source=str(p))

    if len(rows[0]) == 1:
        return SemFeatureVector(y=_float_list([row[0] for row in rows], name="y"), source=str(p))

    if len(rows) != 1:
        raise SemFeatureError("numeric csv without header must be a single row")
    row = _try_float_row(rows[0])
    if row is None:
        raise SemFeatureError("csv row must be numeric")
    return SemFeatureVector(y=_float_list(row, name="y"), source=str(p))


def load_sem_features(source: str | Path | Mapping[str, object] | Sequence[object]) -> SemFeatureVector:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.suffix.lower() == ".json":
            return read_sem_features_json(p)
        if p.suffix.lower() == ".csv":
            return read_sem_features_csv(p)
        raise SemFeatureError(f"unsupported SEM feature format: {p.suffix or '<none>'}")
    return _from_payload(source)
