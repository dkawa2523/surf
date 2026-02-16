from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def normalize_feature_contract(
    payload: Mapping[str, Any],
    *,
    source: str = "feature_contract",
    require_extended: bool = False,
) -> dict[str, Any]:
    names_raw = payload.get("feature_names")
    feat_dim_raw = payload.get("feat_dim")
    cond_dim_raw = payload.get("cond_dim", 0)
    recipe_keys_raw = payload.get("recipe_keys", [])

    if not isinstance(names_raw, Sequence) or isinstance(names_raw, (str, bytes, bytearray)):
        raise ValueError(f"{source}: contract mismatch (feature_names must be a non-empty list)")
    feature_names = [str(v) for v in names_raw]
    if not feature_names:
        raise ValueError(f"{source}: contract mismatch (feature_names must be a non-empty list)")
    feat_dim = int(feat_dim_raw)
    if feat_dim != len(feature_names):
        raise ValueError(f"{source}: contract mismatch (feat_dim does not match feature_names length)")
    cond_dim = max(0, int(cond_dim_raw))
    recipe_keys = [str(v) for v in recipe_keys_raw] if isinstance(recipe_keys_raw, Sequence) and not isinstance(recipe_keys_raw, (str, bytes, bytearray)) else []
    if require_extended and "recipe_keys" not in payload:
        raise ValueError(f"{source}: contract mismatch (recipe_keys must be present)")
    if require_extended and ("band_width" not in payload or "min_grad_norm" not in payload):
        raise ValueError(f"{source}: contract mismatch (band_width and min_grad_norm are required)")
    band_width = float(payload.get("band_width", 0.5))
    min_grad_norm = float(payload.get("min_grad_norm", 1e-6))

    return {
        "feature_names": feature_names,
        "feat_dim": feat_dim,
        "cond_dim": cond_dim,
        "recipe_keys": recipe_keys,
        "band_width": band_width,
        "min_grad_norm": min_grad_norm,
    }


def validate_rows_against_feature_contract(*, rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], source: str = "feature_contract") -> None:
    normalized = normalize_feature_contract(contract, source=source)
    names = [str(name) for name in normalized["feature_names"]]
    expected_dim = int(normalized["feat_dim"])
    if not rows:
        raise ValueError(f"{source}: contract mismatch (rows must be non-empty)")
    name_set = set(names)
    for row_index, row in enumerate(rows):
        ordered_keys = [str(key) for key in row.keys()]
        row_keys = set(ordered_keys)
        missing = [name for name in names if name not in row_keys]
        if missing:
            raise ValueError(
                f"{source}: contract mismatch (row {row_index} missing required columns {missing[:5]})"
            )
        projected = [key for key in ordered_keys if key in name_set]
        if len(projected) != expected_dim:
            raise ValueError(
                f"{source}: contract mismatch (row {row_index} has contract_dim={len(projected)} expected={expected_dim})"
            )
        if projected != names:
            raise ValueError(
                f"{source}: contract mismatch (row {row_index} feature column order differs from contract)"
            )


def load_feature_contract(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"feature_contract path does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, Mapping):
        raise ValueError(f"feature_contract json must be a mapping: {resolved}")
    return normalize_feature_contract(payload, source="feature_contract", require_extended=True)


def assert_feature_contract_compatible(*, expected: Mapping[str, Any], actual: Mapping[str, Any], context: str) -> None:
    lhs = normalize_feature_contract(expected, source=f"{context}.expected", require_extended=True)
    rhs = normalize_feature_contract(actual, source=f"{context}.actual", require_extended=True)
    if lhs["feature_names"] != rhs["feature_names"]:
        raise ValueError(
            f"{context}: contract mismatch (feature_names expected={lhs['feature_names']} actual={rhs['feature_names']})"
        )
    if int(lhs["feat_dim"]) != int(rhs["feat_dim"]):
        raise ValueError(
            f"{context}: contract mismatch (feat_dim expected={lhs['feat_dim']} actual={rhs['feat_dim']})"
        )
    if int(lhs["cond_dim"]) != int(rhs["cond_dim"]):
        raise ValueError(
            f"{context}: contract mismatch (cond_dim expected={lhs['cond_dim']} actual={rhs['cond_dim']})"
        )


def validate_predict_vn_contract(model: Any) -> None:
    if not hasattr(model, "predict_vn"):
        raise TypeError("model contract violation: missing predict_vn(phi, conditions, step_index)")
    fn = getattr(model, "predict_vn")
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    required = {"phi", "conditions"}
    if not required.issubset(set(params)):
        raise TypeError(
            "model contract violation: predict_vn must accept parameters phi and conditions"
        )
    if "step_index" not in params:
        raise TypeError(
            "model contract violation: predict_vn must accept step_index"
        )
