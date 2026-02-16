from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sanitize_run_id(value: str) -> str:
    text = str(value).strip()
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return out or "run"


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"json payload must be a mapping: {path}")
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.is_file():
        raise ValueError(f"csv file does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def read_csv_float_rows(path: Path) -> list[dict[str, float]]:
    rows = read_csv_rows(path)
    out: list[dict[str, float]] = []
    for row in rows:
        out.append({str(key): float(value) for key, value in row.items()})
    return out


def flatten_frame(frame: Any) -> list[float]:
    if hasattr(frame, "tolist"):
        frame = frame.tolist()
    if isinstance(frame, (list, tuple)):
        out: list[float] = []
        for item in frame:
            out.extend(flatten_frame(item))
        return out
    return [float(frame)]


def frame_mean(frame: Any) -> float:
    values = flatten_frame(frame)
    if not values:
        raise ValueError("frame is empty")
    return fmean(values)
