from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python <=3.10
    try:
        import tomli as _toml  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover
        _toml = None  # type: ignore[assignment]

try:
    import yaml as _yaml  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    _yaml = None


class ConfigError(ValueError):
    """Invalid or unsupported configuration file."""


def _as_mapping(path: Path, payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ConfigError(f"config must be a mapping: {path}")
    return {str(key): value for key, value in payload.items()}


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists() or not cfg_path.is_file():
        raise ConfigError(f"config file does not exist: {cfg_path}")

    suffix = cfg_path.suffix.lower()
    if suffix in {".toml", ""}:
        if _toml is None:
            raise ConfigError("TOML parser is unavailable (requires tomllib or tomli)")
        with cfg_path.open("rb") as fp:
            return _as_mapping(cfg_path, _toml.load(fp))

    if suffix in {".yaml", ".yml"}:
        if _yaml is None:
            raise ConfigError("YAML parsing requires optional dependency 'pyyaml'")
        with cfg_path.open("r", encoding="utf-8") as fp:
            return _as_mapping(cfg_path, _yaml.safe_load(fp))

    raise ConfigError(f"unsupported config extension: {suffix or '<none>'}")
