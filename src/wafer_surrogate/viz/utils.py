from __future__ import annotations

import json
import os
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from wafer_surrogate.config.loader import load_config

def is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def load_pyplot() -> Any | None:
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _deep_merge_dict(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        skey = str(key)
        if isinstance(value, Mapping) and isinstance(out.get(skey), dict):
            out[skey] = _deep_merge_dict(dict(out[skey]), value)
        elif isinstance(value, Mapping):
            out[skey] = _deep_merge_dict({}, value)
        else:
            out[skey] = value
    return out


def _to_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def default_visualization_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "defaults": {"enabled": True},
        "train": {
            "learning_curves": True,
            "scatter_gt_pred": True,
            "r2": True,
        },
        "inference": {
            "temporal_diagnostics_plot": True,
        },
        "sdf_views": {
            "xy": True,
            "xz": True,
            "yz": True,
            "gt_pred_error": True,
        },
        "mesh3d": {
            "html": True,
            "cutaway": True,
            "compare_t8": True,
        },
        "export": {
            "dpi": 140,
            "max_frames": "all",
        },
    }


def _load_visualization_yaml(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        warnings.append(f"viz config file not found: {path}")
        return {}
    try:
        payload = load_config(path)
    except Exception as exc:
        warnings.append(f"viz config load failed: {path} ({exc})")
        return {}
    if not isinstance(payload, Mapping):
        warnings.append(f"viz config ignored (not mapping): {path}")
        return {}
    return {str(k): v for k, v in payload.items()}


def resolve_visualization_config(
    *,
    cli_path: str | Path | None = None,
    stage_config: Mapping[str, Any] | None = None,
    run_config: Mapping[str, Any] | None = None,
    default_yaml_path: str | Path | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    warn = warnings if warnings is not None else []
    cfg = default_visualization_config()
    default_yaml = Path(default_yaml_path) if default_yaml_path is not None else Path("configs/visualization.default.yaml")
    cfg = _deep_merge_dict(cfg, _load_visualization_yaml(default_yaml, warn))
    cfg = _deep_merge_dict(cfg, _to_mapping(run_config))
    cfg = _deep_merge_dict(cfg, _to_mapping(stage_config))
    if cli_path is not None and str(cli_path).strip():
        cfg = _deep_merge_dict(cfg, _load_visualization_yaml(Path(str(cli_path)), warn))
    return cfg


def viz_get(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for token in path.split("."):
        if not isinstance(current, Mapping):
            return default
        current = current.get(token)
    if current is None:
        return default
    return current


def viz_enabled(config: Mapping[str, Any], path: str, default: bool = True) -> bool:
    global_default = bool(viz_get(config, "defaults.enabled", True))
    value = viz_get(config, path, global_default if default is True else default)
    return bool(value)


def write_visualization_manifest(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(dict(payload), fp, indent=2)
    return path
