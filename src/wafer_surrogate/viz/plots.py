from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .utils import load_pyplot as _load_pyplot


def _safe_run_id(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value.strip())
    return sanitized or "run"


def _is_plot_artifact(path: Path) -> bool:
    parts = list(path.parts)
    for idx in range(len(parts) - 1):
        if parts[idx] == "viz" and parts[idx + 1] == "plots":
            return True
    return False


def _candidate_metric_files(run_dir: Path) -> list[Path]:
    patterns = (
        "train_summary_*.json",
        "eval_summary_*.json",
        "train_distill_summary_*.json",
        "*metrics*.json",
        "*metrics*.jsonl",
        "*metrics*.csv",
    )
    seen: dict[str, Path] = {}
    for pattern in patterns:
        for path in sorted(run_dir.glob(pattern)):
            if path.is_file() and not _is_plot_artifact(path):
                seen[str(path)] = path
        for path in sorted(run_dir.rglob(pattern)):
            if path.is_file() and not _is_plot_artifact(path):
                seen[str(path)] = path
    return [seen[key] for key in sorted(seen.keys())]


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value_f = float(value)
        return value_f if math.isfinite(value_f) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value_f = float(text)
        except ValueError:
            return None
        return value_f if math.isfinite(value_f) else None
    return None


def _extract_metrics_rows_from_json(path: Path, payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, Mapping):
        metrics = payload.get("metrics")
        if isinstance(metrics, Mapping):
            rows.append(dict(metrics))

        reference_metrics = payload.get("reference_metrics")
        if isinstance(reference_metrics, Mapping):
            rows.append({f"reference_{key}": value for key, value in reference_metrics.items()})

        distill = payload.get("distillation")
        if isinstance(distill, Mapping):
            for key, prefix in (
                ("teacher_metrics", "teacher"),
                ("student_metrics", "student"),
                ("student_distill_metrics", "student_distill"),
            ):
                metric_map = distill.get(key)
                if isinstance(metric_map, Mapping):
                    rows.append({f"{prefix}_{name}": value for name, value in metric_map.items()})

        if not rows and "metrics" in path.name.lower():
            numeric_row: dict[str, float] = {}
            for key, value in payload.items():
                parsed = _to_float(value)
                if parsed is not None:
                    numeric_row[str(key)] = parsed
            if numeric_row:
                rows.append(numeric_row)

        if not rows:
            records = payload.get("records")
            if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
                for row in records:
                    if isinstance(row, Mapping):
                        rows.append(dict(row))

    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for row in payload:
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return rows


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fp:
            return [dict(row) for row in csv.DictReader(fp)]
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if isinstance(payload, Mapping):
                    rows.append(dict(payload))
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return _extract_metrics_rows_from_json(path, payload)
    return []


def _extract_numeric_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in row.items():
        key_l = str(key).strip().lower()
        if key_l in {
            "step",
            "epoch",
            "iter",
            "iteration",
            "trial",
            "time",
            "timestamp",
            "run_id",
            "split",
        }:
            continue
        parsed = _to_float(value)
        if parsed is None:
            continue
        numeric[str(key)] = parsed
    return numeric


def _rank_metrics(series: Mapping[str, Sequence[float]]) -> list[str]:
    return sorted(
        series.keys(),
        key=lambda name: (
            0 if "loss" in name.lower() else 1,
            -len(series[name]),
            name.lower(),
        ),
    )


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    scale = max(abs(float(value)) for value in values)
    if scale == 0.0:
        return 0.0
    return float(scale) * (sum(float(value) / float(scale) for value in values) / float(len(values)))


def _safe_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    scale = max(abs(float(value)) for value in values)
    if scale == 0.0:
        return 0.0
    mean_scaled = sum(float(value) / float(scale) for value in values) / float(len(values))
    var_scaled = sum(((float(value) / float(scale)) - mean_scaled) ** 2 for value in values) / float(len(values))
    return float(scale) * (float(var_scaled) ** 0.5)


def _infer_run_id(run_dir: Path, source_files: Sequence[Path], run_id_hint: str | None) -> str:
    if run_id_hint:
        return _safe_run_id(run_id_hint)
    if run_dir.parent.name == "runs":
        return _safe_run_id(run_dir.name)
    if run_dir.name == "runs":
        for directory in sorted(run_dir.iterdir()):
            if directory.is_dir() and directory.name not in {"logs"}:
                return _safe_run_id(directory.name)
    if source_files:
        first = source_files[0]
        if first.parent.parent.name == "runs":
            return _safe_run_id(first.parent.name)
    return _safe_run_id(run_dir.name)


def _resolve_output_root(run_dir: Path, run_id: str) -> Path:
    if run_dir.parent.name == "runs":
        return run_dir
    if run_dir.name == "runs":
        return run_dir / run_id
    return Path("runs") / run_id


def _plot_curves(plot_path: Path, series: Mapping[str, Sequence[float]], plt: Any) -> None:
    metric_names = _rank_metrics(series)[:8]
    fig, axis = plt.subplots(1, 1, figsize=(9.0, 4.8))
    for metric in metric_names:
        values = [float(value) for value in series[metric]]
        axis.plot(range(1, len(values) + 1), values, label=metric, linewidth=1.5)
    axis.set_title("Loss / Metrics Curves")
    axis.set_xlabel("observation index")
    axis.set_ylabel("value")
    axis.grid(alpha=0.25)
    if metric_names:
        axis.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


def _plot_hist(plot_path: Path, series: Mapping[str, Sequence[float]], plt: Any) -> None:
    metric_names = _rank_metrics(series)[:6]
    if not metric_names:
        return
    cols = min(3, len(metric_names))
    rows = (len(metric_names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
    if hasattr(axes, "flat"):
        axis_list = list(axes.flat)
    elif isinstance(axes, Sequence):
        axis_list = list(axes)
    else:
        axis_list = [axes]
    for axis, metric in zip(axis_list, metric_names):
        values = [float(value) for value in series[metric]]
        bins = max(5, min(20, int(len(values) ** 0.5) + 1))
        axis.hist(values, bins=bins, color="#1f77b4", alpha=0.8)
        axis.set_title(metric)
        axis.set_xlabel("value")
        axis.set_ylabel("count")
    for axis in axis_list[len(metric_names):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


def _r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float | None:
    if not y_true or len(y_true) != len(y_pred):
        return None
    y_true_f = [float(v) for v in y_true]
    y_pred_f = [float(v) for v in y_pred]
    mean_true = _safe_mean(y_true_f)
    ss_tot = sum((value - mean_true) ** 2 for value in y_true_f)
    if abs(ss_tot) <= 1e-12:
        return None
    ss_res = sum((truth - pred) ** 2 for truth, pred in zip(y_true_f, y_pred_f))
    return float(1.0 - (ss_res / ss_tot))


def render_train_output_visuals(
    *,
    output_dir: str | Path,
    metric_files: Sequence[str | Path],
    predictions_csv: str | Path | None,
    learning_curves: bool = True,
    scatter_gt_pred: bool = True,
    r2_enabled: bool = True,
    dpi: int = 140,
) -> dict[str, Any]:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    viz_dir = out_root / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "output_dir": str(out_root),
        "viz_dir": str(viz_dir),
        "learning_curves_enabled": bool(learning_curves),
        "scatter_gt_pred_enabled": bool(scatter_gt_pred),
        "r2_enabled": bool(r2_enabled),
        "sources": {"metric_files": [str(Path(path)) for path in metric_files]},
        "warnings": [],
    }

    series: dict[str, list[float]] = defaultdict(list)
    for raw_path in metric_files:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            manifest["warnings"].append(f"metric file missing: {path}")
            continue
        rows = _load_rows(path)
        for row in rows:
            numeric_metrics = _extract_numeric_metrics(row)
            for metric, value in sorted(numeric_metrics.items()):
                series[metric].append(float(value))

    plt = _load_pyplot()
    if plt is None:
        manifest["warnings"].append("matplotlib unavailable")

    learning_curves_path: Path | None = None
    if learning_curves:
        if plt is None:
            manifest["warnings"].append("learning_curves skipped: matplotlib unavailable")
        elif not series:
            manifest["warnings"].append("learning_curves skipped: no numeric metrics")
        else:
            learning_curves_path = viz_dir / "learning_curves.png"
            _plot_curves(learning_curves_path, series, plt)

    scatter_path: Path | None = None
    r2_path: Path | None = None
    if scatter_gt_pred or r2_enabled:
        records: list[dict[str, Any]] = []
        if predictions_csv is None:
            manifest["warnings"].append("scatter/r2 skipped: predictions csv missing")
        else:
            pred_path = Path(predictions_csv)
            if not pred_path.exists() or not pred_path.is_file():
                manifest["warnings"].append(f"scatter/r2 skipped: predictions csv not found ({pred_path})")
            else:
                with pred_path.open("r", encoding="utf-8", newline="") as fp:
                    for row in csv.DictReader(fp):
                        records.append(dict(row))

        if records:
            grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
            for row in records:
                split = str(row.get("split", "unknown"))
                y_true = _to_float(row.get("y_true"))
                y_pred = _to_float(row.get("y_pred"))
                if y_true is None or y_pred is None:
                    continue
                grouped[split]["y_true"].append(float(y_true))
                grouped[split]["y_pred"].append(float(y_pred))

            r2_payload: dict[str, Any] = {"overall": None, "by_split": {}}
            all_true = [v for group in grouped.values() for v in group["y_true"]]
            all_pred = [v for group in grouped.values() for v in group["y_pred"]]
            overall = _r2_score(all_true, all_pred)
            r2_payload["overall"] = overall
            for split, group in sorted(grouped.items()):
                r2_payload["by_split"][split] = _r2_score(group["y_true"], group["y_pred"])

            if r2_enabled:
                r2_path = viz_dir / "train_r2.json"
                with r2_path.open("w", encoding="utf-8") as fp:
                    json.dump(r2_payload, fp, indent=2)

            if scatter_gt_pred:
                if plt is None:
                    manifest["warnings"].append("scatter plot skipped: matplotlib unavailable")
                else:
                    scatter_path = viz_dir / "scatter_gt_pred.png"
                    fig, axis = plt.subplots(1, 1, figsize=(6.2, 5.2))
                    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]
                    for idx, (split, group) in enumerate(sorted(grouped.items())):
                        if not group["y_true"]:
                            continue
                        axis.scatter(
                            group["y_true"],
                            group["y_pred"],
                            alpha=0.7,
                            s=14,
                            label=f"{split} (n={len(group['y_true'])})",
                            color=palette[idx % len(palette)],
                        )
                    if all_true and all_pred:
                        all_values = all_true + all_pred
                        lo = min(all_values)
                        hi = max(all_values)
                        if hi > lo:
                            axis.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="#555555")
                    r2_note = f"R2={overall:.4f}" if isinstance(overall, float) else "R2=NA"
                    axis.set_title(f"GT vs PRED Scatter ({r2_note})")
                    axis.set_xlabel("y_true")
                    axis.set_ylabel("y_pred")
                    axis.grid(alpha=0.25)
                    axis.legend(loc="best", fontsize=8)
                    fig.tight_layout()
                    fig.savefig(scatter_path, dpi=max(72, int(dpi)))
                    plt.close(fig)
        else:
            manifest["warnings"].append("scatter/r2 skipped: no prediction rows")

    manifest["learning_curves_path"] = None if learning_curves_path is None else str(learning_curves_path)
    manifest["scatter_gt_pred_path"] = None if scatter_path is None else str(scatter_path)
    manifest["r2_path"] = None if r2_path is None else str(r2_path)
    manifest_path = viz_dir / "visualization_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    return {
        "viz_dir": viz_dir,
        "manifest_path": manifest_path,
        "learning_curves_path": learning_curves_path,
        "scatter_gt_pred_path": scatter_path,
        "r2_path": r2_path,
    }


def plot_metrics_for_run_dir(
    *,
    run_dir: str | Path,
    run_id_hint: str | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    if not run_path.exists() or not run_path.is_dir():
        raise ValueError(f"run-dir does not exist: {run_path}")

    source_files = _candidate_metric_files(run_path)
    series: dict[str, list[float]] = defaultdict(list)
    points: list[dict[str, Any]] = []
    for source in source_files:
        rows = _load_rows(source)
        for row in rows:
            numeric_metrics = _extract_numeric_metrics(row)
            for metric, value in sorted(numeric_metrics.items()):
                series[metric].append(float(value))
                points.append(
                    {
                        "metric": metric,
                        "point_index": len(series[metric]),
                        "value": float(value),
                        "source_file": str(source),
                    }
                )

    run_id = _infer_run_id(run_path, source_files, run_id_hint)
    output_root = _resolve_output_root(run_path, run_id=run_id)
    plots_dir = output_root / "viz" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    points_csv = plots_dir / "metrics_points.csv"
    with points_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["metric", "point_index", "value", "source_file"])
        writer.writeheader()
        for row in points:
            writer.writerow(row)

    hist_csv = plots_dir / "metrics_hist_summary.csv"
    with hist_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["metric", "count", "min", "max", "mean", "std"])
        writer.writeheader()
        for metric in _rank_metrics(series):
            values = [float(value) for value in series[metric]]
            writer.writerow(
                {
                    "metric": metric,
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": _safe_mean(values),
                    "std": _safe_std(values),
                }
            )

    plt = _load_pyplot()
    curves_png: Path | None = None
    hist_png: Path | None = None
    if plt is not None and series:
        curves_png = plots_dir / "loss_metrics_curves.png"
        hist_png = plots_dir / "metrics_hist.png"
        _plot_curves(curves_png, series, plt)
        _plot_hist(hist_png, series, plt)

    manifest_path = plots_dir / "plot_manifest.json"
    manifest = {
        "run_dir": str(run_path),
        "output_root": str(output_root),
        "plots_dir": str(plots_dir),
        "matplotlib_available": plt is not None,
        "num_source_files": len(source_files),
        "num_metrics": len(series),
        "num_points": len(points),
        "source_files": [str(path) for path in source_files],
        "points_csv_path": str(points_csv),
        "hist_summary_csv_path": str(hist_csv),
        "curves_png_path": None if curves_png is None else str(curves_png),
        "hist_png_path": None if hist_png is None else str(hist_png),
    }
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    return {
        "run_id": run_id,
        "run_dir": run_path,
        "output_root": output_root,
        "plots_dir": plots_dir,
        "matplotlib_available": plt is not None,
        "num_source_files": len(source_files),
        "num_metrics": len(series),
        "num_points": len(points),
        "points_csv_path": points_csv,
        "hist_summary_csv_path": hist_csv,
        "curves_png_path": curves_png,
        "hist_png_path": hist_png,
        "manifest_path": manifest_path,
    }
