from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from wafer_surrogate.config.loader import load_config as load_config_file
from wafer_surrogate.core import (
    frame_mean as core_frame_mean,
    frame_to_list,
    now_utc as core_now_utc,
    rollout as core_rollout,
    sanitize_run_id as core_sanitize_run_id,
    to_float_map as core_to_float_map,
)
from wafer_surrogate.data.synthetic import SyntheticSDFDataset, SyntheticSDFRun, generate_synthetic_sdf_dataset
from wafer_surrogate.features import make_feature_extractor
from wafer_surrogate.metrics import compute_rollout_metrics
from wafer_surrogate.models import make_model
from wafer_surrogate.preprocess import build_preprocess_pipeline
from wafer_surrogate.workflows.prior_utils import build_prior as workflow_build_prior
from wafer_surrogate.workflows.prior_utils import prior_preview as workflow_prior_preview

MODEL_ALIASES = {
    "baseline": "baseline_vn_constant",
    "operator": "operator_time_conditioned",
}


def now_utc() -> str:
    return core_now_utc()


def load_config(path: Path) -> dict[str, Any]:
    return load_config_file(path)


def output_dir(config: Mapping[str, Any]) -> Path:
    run_cfg = config.get("run", {})
    if not isinstance(run_cfg, Mapping):
        raise ValueError("config[run] must be a table")
    out_dir = Path(str(run_cfg.get("output_dir", "runs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_dataset(config: Mapping[str, Any]) -> SyntheticSDFDataset:
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise ValueError("config[data] must be a table")
    source = str(data_cfg.get("source", "synthetic"))
    if source != "synthetic":
        raise ValueError(f"unsupported data.source='{source}' in P0; use 'synthetic'")
    return generate_synthetic_sdf_dataset(
        num_runs=int(data_cfg.get("num_runs", 1)),
        num_steps=int(data_cfg.get("num_steps", 6)),
        grid_size=int(data_cfg.get("grid_size", 24)),
        dt=float(data_cfg.get("dt", 0.1)),
        dimension=int(data_cfg.get("dimension", 2)),
        grid_depth=int(data_cfg.get("grid_depth", 16)),
    )


def build_model(config: Mapping[str, Any]) -> tuple[str, dict[str, Any], Any]:
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, Mapping):
        raise ValueError("config[model] must be a table")
    requested = str(model_cfg.get("name", "baseline_vn_constant"))
    model_name = MODEL_ALIASES.get(requested, requested)
    kwargs: dict[str, Any] = {}
    if "default_value" in model_cfg:
        kwargs["default_value"] = float(model_cfg["default_value"])
    elif "default_vn" in model_cfg:
        kwargs["default_value"] = float(model_cfg["default_vn"])
    weights = model_cfg.get("condition_weights")
    if isinstance(weights, Mapping):
        kwargs["condition_weights"] = {str(k): float(v) for k, v in weights.items()}
    return model_name, kwargs, make_model(model_name, **kwargs)


def build_prior(config: Mapping[str, Any]) -> tuple[str, dict[str, Any], Any]:
    return workflow_build_prior(config)


def prior_preview(prior: Any) -> dict[str, Any]:
    return workflow_prior_preview(prior)


def inference_conditions(config: Mapping[str, Any], fallback: Mapping[str, float]) -> dict[str, float]:
    infer_cfg = config.get("inference", {})
    if not isinstance(infer_cfg, Mapping):
        return {str(key): float(value) for key, value in fallback.items()}
    cond = infer_cfg.get("conditions")
    if isinstance(cond, Mapping):
        return {str(key): float(value) for key, value in cond.items()}
    return {str(key): float(value) for key, value in fallback.items()}


def ood_threshold(config: Mapping[str, Any]) -> float:
    infer_cfg = config.get("inference", {})
    if not isinstance(infer_cfg, Mapping):
        return 3.0
    threshold = infer_cfg.get("ood_distance_threshold", 3.0)
    return float(threshold)


def fit_model(model: Any, dataset: SyntheticSDFDataset) -> dict[str, float]:
    if hasattr(model, "fit_operator") and hasattr(model, "predict_phi"):
        model.fit_operator(dataset.runs)
        frame_errors: list[float] = []
        num_frames = 0
        for run in dataset.runs:
            phi0 = run.phi_t[0]
            for i, frame in enumerate(run.phi_t):
                pred = to_list(
                    model.predict_phi(
                        phi0=phi0,
                        conditions=run.recipe,
                        t=float(i) * float(run.dt),
                    )
                )
                ref = to_list(frame)
                for pred_row, ref_row in zip(pred, ref):
                    for pred_cell, ref_cell in zip(pred_row, ref_row):
                        frame_errors.append(abs(float(pred_cell) - float(ref_cell)))
                num_frames += 1
        if not frame_errors:
            raise ValueError("operator model did not produce frame predictions")
        return {
            "num_samples": float(num_frames),
            "target_mean": 0.0,
            "target_std": 0.0,
            "mae": fmean(frame_errors),
        }

    features: list[dict[str, float]] = []
    targets: list[float] = []
    for run in dataset.runs:
        for i in range(len(run.phi_t) - 1):
            prev_vals = [float(cell) for row in run.phi_t[i] for cell in row]
            next_vals = [float(cell) for row in run.phi_t[i + 1] for cell in row]
            targets.append((fmean(prev_vals) - fmean(next_vals)) / float(run.dt))
            features.append({str(k): float(v) for k, v in run.recipe.items()})
    if not targets:
        raise ValueError("synthetic dataset did not produce train targets")
    model.fit(features, targets)
    preds = [float(model.predict(sample)) for sample in features]
    errors = [abs(pred - target) for pred, target in zip(preds, targets)]
    return {
        "num_samples": float(len(targets)),
        "target_mean": fmean(targets),
        "target_std": pstdev(targets) if len(targets) > 1 else 0.0,
        "mae": fmean(errors),
    }


def write_json_timestamped(out_dir: Path, stem: str, payload: Mapping[str, Any]) -> Path:
    path = out_dir / f"{stem}_{now_utc()}.json"
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return path


def write_csv_timestamped(
    out_dir: Path,
    stem: str,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = out_dir / f"{stem}_{now_utc()}.csv"
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return path


def to_list(frame: Any) -> list[list[float]]:
    return frame_to_list(frame)


def rollout(run: SyntheticSDFRun, model: Any) -> list[list[list[float]]]:
    return [to_list(frame) for frame in core_rollout(run, model)]


def eval_metrics(predicted_runs: Sequence[list[list[list[float]]]], dataset: SyntheticSDFDataset) -> dict[str, float]:
    return compute_rollout_metrics(predicted_runs=predicted_runs, reference_runs=dataset.runs)


def split_eval_samples(
    dataset: SyntheticSDFDataset,
    predicted_runs: Sequence[list[list[list[float]]]],
) -> dict[str, list[tuple[SyntheticSDFRun, list[list[list[float]]]]]]:
    if len(predicted_runs) != len(dataset.runs):
        raise ValueError("predicted_runs length must match dataset.runs length")
    paired = [(dataset.runs[idx], predicted_runs[idx]) for idx in range(len(dataset.runs))]
    if not paired:
        return {"val": [], "test": []}
    midpoint = max(1, len(paired) // 2)
    val_samples = paired[:midpoint]
    test_samples = paired[midpoint:] or paired[:midpoint]
    return {"val": val_samples, "test": test_samples}


def write_eval_compare_sections(
    *,
    out_dir: Path,
    split_samples: Mapping[str, Sequence[tuple[SyntheticSDFRun, list[list[list[float]]]]]],
    max_samples: int,
    y_index: int | None,
    contour_level: float = 0.0,
) -> dict[str, Any]:
    from wafer_surrogate.viz.compare_sections import render_compare_section

    records: list[dict[str, Any]] = []
    capped = max(0, int(max_samples))
    for split_name, pairs in split_samples.items():
        for sample_index, (gt_run, pred_phi_t) in enumerate(list(pairs)[:capped]):
            if not pred_phi_t or not gt_run.phi_t:
                continue
            compare_dir = out_dir / safe_run_id(gt_run.run_id) / "viz" / "png" / "compare"
            result = render_compare_section(
                pred_field=pred_phi_t[-1],
                gt_field=gt_run.phi_t[-1],
                out_dir=compare_dir,
                split=split_name,
                sample_index=sample_index,
                y_index=y_index,
                contour_level=float(contour_level),
                frame_index=len(gt_run.phi_t) - 1,
            )
            records.append(
                {
                    "split": split_name,
                    "run_id": gt_run.run_id,
                    "out_dir": str(result["out_dir"]),
                    "png_path": None if result["png_path"] is None else str(result["png_path"]),
                    "fallback_path": None if result["fallback_path"] is None else str(result["fallback_path"]),
                    "matplotlib_available": bool(result["matplotlib_available"]),
                    "y_index": int(result["y_index"]),
                }
            )
    return {
        "enabled": True,
        "requested_max_samples": capped,
        "generated": len(records),
        "records": records,
    }


def write_plot(out_dir: Path, phi_t: Sequence[Sequence[Sequence[float]]]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].imshow(phi_t[0], cmap="viridis")
    axes[1].imshow(phi_t[-1], cmap="viridis")
    fig.tight_layout()
    plot_path = out_dir / f"infer_plot_{now_utc()}.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def to_float_map(payload: Mapping[str, float]) -> dict[str, float]:
    return core_to_float_map(payload)


def maps_equal(lhs: Mapping[str, float], rhs: Mapping[str, float], tol: float = 1e-12) -> bool:
    if set(lhs) != set(rhs):
        return False
    return all(abs(float(lhs[key]) - float(rhs[key])) <= tol for key in lhs)


def frame_mean(frame: Sequence[Sequence[float]]) -> float:
    return core_frame_mean(frame)


def batch_conditions(
    config: Mapping[str, Any],
    fallback: Mapping[str, float],
    batch_size: int,
) -> list[dict[str, float]]:
    infer_cfg = config.get("inference", {})
    if isinstance(infer_cfg, Mapping):
        configured = infer_cfg.get("batch_conditions")
        if isinstance(configured, Sequence):
            parsed = [
                to_float_map(item)
                for item in configured
                if isinstance(item, Mapping)
            ]
            if parsed:
                return parsed
    base = to_float_map(fallback)
    if batch_size <= 1:
        return [base]
    centered_offset = (batch_size - 1) / 2.0
    keys = sorted(base.keys())
    conditions: list[dict[str, float]] = []
    for index in range(batch_size):
        shift = float(index) - centered_offset
        condition: dict[str, float] = {}
        for key in keys:
            baseline = float(base[key])
            delta_scale = max(abs(baseline) * 0.05, 0.02)
            condition[key] = baseline + delta_scale * shift
        conditions.append(condition)
    return conditions


def search_ranges(
    config: Mapping[str, Any],
    fallback: Mapping[str, float],
) -> dict[str, tuple[float, float]]:
    search_cfg = config.get("search", {})
    if isinstance(search_cfg, Mapping):
        ranges_cfg = search_cfg.get("ranges")
        if isinstance(ranges_cfg, Mapping):
            ranges: dict[str, tuple[float, float]] = {}
            for key, raw_range in ranges_cfg.items():
                if (
                    isinstance(raw_range, Sequence)
                    and not isinstance(raw_range, (str, bytes))
                    and len(raw_range) == 2
                ):
                    lo = float(raw_range[0])
                    hi = float(raw_range[1])
                    ranges[str(key)] = (min(lo, hi), max(lo, hi))
            if ranges:
                return ranges
    inferred_ranges: dict[str, tuple[float, float]] = {}
    for key, value in to_float_map(fallback).items():
        radius = max(abs(value) * 0.2, 0.05)
        inferred_ranges[key] = (value - radius, value + radius)
    return inferred_ranges


def flatten_condition_row(
    row: Mapping[str, Any],
    condition_keys: Sequence[str],
) -> dict[str, Any]:
    flat = {key: row.get(key) for key in row if key != "conditions"}
    cond = row.get("conditions")
    cond_map = cond if isinstance(cond, Mapping) else {}
    for key in condition_keys:
        flat[f"cond_{key}"] = float(cond_map.get(key, 0.0))
    return flat


def make_samples_with_pipeline(
    dataset: SyntheticSDFDataset,
    extractor_name: str,
    preprocess_specs: Sequence[tuple[str, Mapping[str, object]]],
) -> tuple[list[dict[str, float]], list[float]]:
    extractor = make_feature_extractor(extractor_name)
    pipeline = build_preprocess_pipeline(preprocess_specs)
    features: list[dict[str, float]] = []
    targets: list[float] = []
    for run in dataset.runs:
        extracted = extractor.extract(run.recipe)
        processed = pipeline.transform(extracted)
        for i in range(len(run.phi_t) - 1):
            prev_vals = [float(cell) for row in run.phi_t[i] for cell in row]
            next_vals = [float(cell) for row in run.phi_t[i + 1] for cell in row]
            targets.append((fmean(prev_vals) - fmean(next_vals)) / float(run.dt))
            features.append({str(key): float(value) for key, value in processed.items()})
    if not targets:
        raise ValueError("synthetic dataset did not produce train targets")
    return features, targets


def safe_run_id(value: str) -> str:
    return core_sanitize_run_id(value)


def write_smoke_metric_logs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = run_dir / "metrics.jsonl"
    with metrics_jsonl.open("w", encoding="utf-8") as fp:
        for epoch in range(6):
            loss = max(0.0, 0.26 - (0.04 * float(epoch)))
            val_mae = 0.19 - (0.02 * float(epoch))
            fp.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": loss,
                        "val_mae": val_mae,
                        "val_rmse": val_mae * 1.2,
                    }
                )
            )
            fp.write("\n")

    eval_csv = run_dir / "eval_metrics.csv"
    with eval_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["step", "sdf_l1_mean", "obs_feature_mae"])
        writer.writeheader()
        for step in range(6):
            writer.writerow(
                {
                    "step": step,
                    "sdf_l1_mean": 0.11 - (0.01 * float(step)),
                    "obs_feature_mae": 0.09 - (0.007 * float(step)),
                }
            )

    eval_json = run_dir / "eval_metrics.json"
    with eval_json.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "metrics": {
                    "sdf_l1_mean": 0.06,
                    "sdf_l2_rmse": 0.08,
                    "obs_feature_mae": 0.04,
                }
            },
            fp,
            indent=2,
        )
