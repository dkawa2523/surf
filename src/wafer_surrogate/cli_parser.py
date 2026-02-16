from __future__ import annotations

import argparse
from typing import Any

from wafer_surrogate.cli_commands.workflow import (
    cmd_eval,
    cmd_infer,
    cmd_infer_batch,
    cmd_pipeline_run,
    cmd_search_recipe,
    cmd_sweep_pipelines,
    cmd_train,
    cmd_train_distill,
)
from wafer_surrogate.cli_commands.viz import (
    cmd_viz_compare_sections,
    cmd_viz_export_vti,
    cmd_viz_leaderboard,
    cmd_viz_plot_metrics,
    cmd_viz_render_slices,
)


def _set_command_handler(parser: argparse.ArgumentParser, handler: Any) -> None:
    parser.set_defaults(handler=handler, func=handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wafer-surrogate", description="CLI entrypoint for wafer surrogate workflows.")
    subparsers = parser.add_subparsers(dest="command")
    train = subparsers.add_parser("train", help="Train baseline model on synthetic data.")
    train_distill = subparsers.add_parser(
        "train-distill",
        help="(Deprecated; remove_after=2026-06-30) Train teacher/student distillation; internally delegates to pipeline train(mode=sparse_distill).",
    )
    infer = subparsers.add_parser("infer", help="Run inference and write outputs under runs/.")
    eval_parser = subparsers.add_parser("eval", help="Evaluate rollout against synthetic references.")
    infer_batch = subparsers.add_parser("infer-batch", help="Run batched inference and write summary artifacts.")
    search_recipe = subparsers.add_parser("search-recipe", help="Run recipe optimization with random/BO/MFBO strategies.")
    sweep_pipelines = subparsers.add_parser("sweep-pipelines", help="Compare feature/preprocess/model pipeline combos.")
    pipeline = subparsers.add_parser(
        "pipeline",
        help="Run composable stage pipeline (use: pipeline run ...).",
    )
    viz = subparsers.add_parser("viz", help="Visualization/export commands.")
    for sub, handler in (
        (train, cmd_train),
        (train_distill, cmd_train_distill),
        (infer, cmd_infer),
        (eval_parser, cmd_eval),
        (infer_batch, cmd_infer_batch),
        (search_recipe, cmd_search_recipe),
        (sweep_pipelines, cmd_sweep_pipelines),
    ):
        sub.add_argument("--config", default="configs/example.toml", help="Path to TOML config.")
        _set_command_handler(sub, handler)
    train.add_argument("--smoke", action="store_true", help="Use a small synthetic dataset for smoke train.")
    eval_parser.add_argument("--smoke", action="store_true", help="Use a small synthetic dataset for smoke eval.")
    eval_parser.add_argument(
        "--compare-sections",
        dest="compare_sections",
        action="store_true",
        help="Generate pred/gt XZ contour comparison artifacts during eval.",
    )
    eval_parser.add_argument(
        "--no-compare-sections",
        dest="compare_sections",
        action="store_false",
        help="Disable pred/gt XZ contour comparison artifact generation.",
    )
    eval_parser.add_argument(
        "--compare-max-samples",
        type=int,
        default=3,
        help="Max samples per split for compare-sections (default: 3).",
    )
    eval_parser.add_argument("--y-index", type=int, help="Fixed y-index for compare-sections (default: center).")
    eval_parser.add_argument(
        "--plot-metrics",
        dest="plot_metrics",
        action="store_true",
        help="Run viz plot-metrics automatically after eval.",
    )
    eval_parser.add_argument(
        "--no-plot-metrics",
        dest="plot_metrics",
        action="store_false",
        help="Disable automatic viz plot-metrics after eval.",
    )
    eval_parser.set_defaults(compare_sections=True, plot_metrics=True)
    infer.add_argument("--plot", dest="plot", action="store_true", help="Write plot artifact.")
    infer.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plotting.")
    infer.set_defaults(plot=False)
    infer_batch.add_argument("--batch-size", type=int, default=4, help="Number of conditions to evaluate in batch mode.")
    search_recipe.add_argument("--trials", type=int, default=20, help="Optimization trial count.")
    search_recipe.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    search_recipe.add_argument(
        "--strategy",
        choices=["random", "grid", "bo", "mfbo"],
        default="random",
        help="Search strategy.",
    )
    search_recipe.add_argument(
        "--engine",
        default="builtin",
        help="Optimization engine: builtin | optuna | plugin:<module[:function]>",
    )
    search_recipe.add_argument(
        "--optuna-sampler",
        choices=["tpe", "random"],
        default="tpe",
        help="Sampler when --engine optuna is used.",
    )
    search_recipe.add_argument(
        "--bo-candidates",
        type=int,
        default=64,
        help="Candidate pool size per BO iteration.",
    )
    search_recipe.add_argument(
        "--mfbo-pool-size",
        type=int,
        default=12,
        help="Low-fidelity screening pool size per MFBO iteration.",
    )
    search_recipe.add_argument(
        "--mfbo-top-k",
        type=int,
        default=3,
        help="Top-k low-fidelity candidates promoted to BO acquisition in MFBO.",
    )
    pipeline_subparsers = pipeline.add_subparsers(dest="pipeline_command")
    pipeline_run = pipeline_subparsers.add_parser("run", help="Run stage pipeline with optional subset.")
    pipeline_run.add_argument("--config", default="configs/example.toml", help="Path to TOML/YAML config.")
    pipeline_run.add_argument(
        "--stages",
        help="Comma-separated stage list, e.g. cleaning,featurization,preprocessing,train,inference",
    )
    _set_command_handler(pipeline_run, cmd_pipeline_run)
    viz_subparsers = viz.add_subparsers(dest="viz_command")
    viz_export_vti = viz_subparsers.add_parser("export-vti", help="Export phi(t) to VTI/PVD for ParaView.")
    viz_export_vti.add_argument("--run-dir", help="Run directory or runs root containing infer/synthetic JSON outputs.")
    viz_export_vti.add_argument("--phi-npz", help="Path to npz with phi_t[time,...] (requires numpy).")
    viz_export_vti.add_argument("--smoke", action="store_true", help="Generate synthetic smoke data and export it.")
    _set_command_handler(viz_export_vti, cmd_viz_export_vti)
    viz_render_slices = viz_subparsers.add_parser("render-slices", help="Render XY/XZ/YZ quicklook slice PNGs.")
    viz_render_slices.add_argument("--run-dir", help="Run directory or runs root containing infer/synthetic JSON outputs.")
    viz_render_slices.add_argument("--phi-npz", help="Path to npz with phi_t[time,...] (requires numpy).")
    viz_render_slices.add_argument("--vti-dir", help="Directory containing phi_t*.vti files.")
    viz_render_slices.add_argument("--smoke", action="store_true", help="Generate synthetic smoke data and render it.")
    viz_render_slices.add_argument("--x-index", type=int, help="Slice position along x-axis (default: center).")
    viz_render_slices.add_argument("--y-index", type=int, help="Slice position along y-axis (default: center).")
    viz_render_slices.add_argument("--z-index", type=int, help="Slice position along z-axis (default: center).")
    viz_render_slices.add_argument(
        "--contour-level",
        type=float,
        default=0.0,
        help="Contour level to overlay (default: 0.0).",
    )
    _set_command_handler(viz_render_slices, cmd_viz_render_slices)
    viz_compare_sections = viz_subparsers.add_parser(
        "compare-sections",
        help="Render XZ contour overlay (pred vs gt) with matplotlib fallback.",
    )
    viz_compare_sections.add_argument("--smoke", action="store_true", help="Generate synthetic smoke compare output.")
    viz_compare_sections.add_argument("--split", default="val", help="Split label in output file names.")
    viz_compare_sections.add_argument("--y-index", type=int, help="Slice position along y-axis (default: center).")
    viz_compare_sections.add_argument(
        "--contour-level",
        type=float,
        default=0.0,
        help="Contour level to compare (default: 0.0).",
    )
    _set_command_handler(viz_compare_sections, cmd_viz_compare_sections)
    viz_plot_metrics = viz_subparsers.add_parser(
        "plot-metrics",
        help="Generate loss/metrics curves + histogram artifacts from run logs.",
    )
    viz_plot_metrics.add_argument("--run-dir", help="Run directory or runs root containing metrics logs.")
    viz_plot_metrics.add_argument("--smoke", action="store_true", help="Generate synthetic smoke metrics and plot them.")
    _set_command_handler(viz_plot_metrics, cmd_viz_plot_metrics)
    viz_leaderboard = viz_subparsers.add_parser(
        "leaderboard",
        help="Render leaderboard score plots and fallback CSV summary.",
    )
    viz_leaderboard.add_argument(
        "--run-dir",
        required=True,
        help="Pipeline run directory (e.g. runs/<run_id>) that contains leaderboard/.",
    )
    _set_command_handler(viz_leaderboard, cmd_viz_leaderboard)
    return parser
