from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .utils import load_pyplot as _load_pyplot


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def _to_float(value: Any) -> float | None:
    try:
        if isinstance(value, bool):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _parse_board_rows(rows: Sequence[Mapping[str, Any]], board: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for row in rows:
        score = _to_float(row.get("score"))
        if score is None:
            continue
        name = str(row.get("name", f"{board}_{len(parsed)}"))
        entry: dict[str, Any] = {
            "board": board,
            "name": name,
            "score": score,
        }
        if row.get("primary_metric") is not None:
            entry["primary_metric"] = str(row.get("primary_metric"))
        if row.get("secondary_metric") is not None:
            entry["secondary_metric"] = str(row.get("secondary_metric"))
        secondary_score = _to_float(row.get("secondary_score"))
        if secondary_score is not None:
            entry["secondary_score"] = secondary_score
        for key, value in row.items():
            if str(key).startswith("metric_"):
                parsed_value = _to_float(value)
                if parsed_value is not None:
                    entry[str(key)] = parsed_value
        parsed.append(entry)
    return parsed


def _plot_scores(path: Path, rows: Sequence[Mapping[str, Any]], plt: Any) -> None:
    boards = sorted({str(row.get("board", "board")) for row in rows})
    if not boards:
        return
    fig, axes = plt.subplots(1, len(boards), figsize=(6.2 * len(boards), 4.2))
    if hasattr(axes, "flat"):
        axis_list = list(axes.flat)
    elif isinstance(axes, Sequence):
        axis_list = list(axes)
    else:
        axis_list = [axes]

    for axis, board in zip(axis_list, boards):
        board_rows = [row for row in rows if str(row.get("board")) == board]
        board_rows = sorted(
            board_rows,
            key=lambda item: (
                float(item.get("metric_student_mae", item.get("score", 1e99))),
                float(item.get("metric_rollout_short_window_error", item.get("metric_rmse", 1e99))),
                float(item.get("score", 1e99)),
            ),
        )[:12]
        labels = [str(row.get("name")) for row in board_rows]
        values = [float(row.get("score", 0.0)) for row in board_rows]
        y = list(range(len(labels)))
        axis.barh(y, values, color="#1f77b4", alpha=0.85)
        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("score")
        axis.set_title(f"{board} leaderboard")
        axis.grid(alpha=0.22, axis="x")

    for axis in axis_list[len(boards):]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_mae_rmse(path: Path, rows: Sequence[Mapping[str, Any]], plt: Any) -> bool:
    points = []
    for row in rows:
        mae = _to_float(row.get("metric_mae"))
        rmse = _to_float(row.get("metric_rmse"))
        if mae is None or rmse is None:
            continue
        points.append((str(row.get("name", "candidate")), mae, rmse))
    if not points:
        return False

    fig, axis = plt.subplots(1, 1, figsize=(7.0, 5.0))
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    axis.scatter(xs, ys, c="#d62728", alpha=0.88)
    for name, x, y in points[:20]:
        axis.annotate(name, (x, y), fontsize=7)
    axis.set_xlabel("MAE")
    axis.set_ylabel("RMSE")
    axis.set_title("Leaderboard: MAE vs RMSE")
    axis.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def render_leaderboard_for_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    board_root = run_path / "leaderboard"
    data_csv = board_root / "data_path" / "leaderboard.csv"
    model_csv = board_root / "model_path" / "leaderboard.csv"

    rows = _parse_board_rows(_read_csv_rows(data_csv), "data_path") + _parse_board_rows(
        _read_csv_rows(model_csv),
        "model_path",
    )

    viz_dir = board_root / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    long_csv = viz_dir / "leaderboard_long.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["board", "name", "score"]
    with long_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))

    plt = _load_pyplot()
    score_png: Path | None = None
    scatter_png: Path | None = None
    if plt is not None and rows:
        score_png = viz_dir / "leaderboard_scores.png"
        _plot_scores(score_png, rows, plt)
        scatter_candidate = viz_dir / "leaderboard_mae_rmse.png"
        if _plot_mae_rmse(scatter_candidate, rows, plt):
            scatter_png = scatter_candidate

    summary = {
        "run_dir": str(run_path),
        "leaderboard_dir": str(board_root),
        "viz_dir": str(viz_dir),
        "num_records": len(rows),
        "matplotlib_available": plt is not None,
        "long_csv_path": str(long_csv),
        "score_png_path": None if score_png is None else str(score_png),
        "scatter_png_path": None if scatter_png is None else str(scatter_png),
        "source_csv": {
            "data_path": str(data_csv),
            "model_path": str(model_csv),
        },
        "ranking_priority": ["metric_student_mae", "metric_rollout_short_window_error", "score"],
    }
    summary_path = viz_dir / "leaderboard_plot_manifest.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    return {
        **summary,
        "summary_path": str(summary_path),
    }
