from __future__ import annotations

from pathlib import Path
from typing import Any

from wafer_surrogate.core.io_utils import (
    flatten_frame,
    frame_mean,
    now_utc,
    sanitize_run_id,
    write_csv,
    write_json,
)


def ensure_stage_dirs(run_dir: Path, stage_name: str) -> dict[str, Path]:
    stage_root = run_dir / stage_name
    dirs = {
        "root": stage_root,
        "configuration": stage_root / "configuration",
        "logs": stage_root / "logs",
        "outputs": stage_root / "outputs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
