from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SDFGrid = list[list[float]]
SDFVolume = list[list[list[float]]]


@dataclass(frozen=True)
class SyntheticSDFRun:
    run_id: str
    dt: float
    recipe: dict[str, float]
    phi_t: list[object]

    def to_dict(self) -> dict[str, object]:
        def _frame_to_list(frame: object) -> object:
            if hasattr(frame, "tolist"):
                return frame.tolist()
            return frame

        return {
            "run_id": self.run_id,
            "dt": self.dt,
            "recipe": dict(self.recipe),
            "phi_t": [_frame_to_list(frame) for frame in self.phi_t],
        }


@dataclass(frozen=True)
class SyntheticSDFDataset:
    runs: list[SyntheticSDFRun]

    def to_dict(self) -> dict[str, object]:
        return {"runs": [run.to_dict() for run in self.runs]}


def _normalize_argv(argv: Sequence[str] | None) -> list[str]:
    args = list(argv) if argv is not None else []
    if args[:1] == ["--"]:
        return args[1:]
    return args


def _circle_sdf(x: float, y: float, cx: float, cy: float, radius: float) -> float:
    return math.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius


def _sphere_sdf(x: float, y: float, z: float, cx: float, cy: float, cz: float, radius: float) -> float:
    return math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - radius


def _build_phi_frame(grid_size: int, radius: float) -> SDFGrid:
    center = 0.5 * (grid_size - 1)
    frame: SDFGrid = []
    for y in range(grid_size):
        row = []
        for x in range(grid_size):
            row.append(_circle_sdf(float(x), float(y), center, center, radius))
        frame.append(row)
    return frame


def _build_phi_volume(grid_size: int, grid_depth: int, radius: float) -> SDFVolume:
    cx = 0.5 * (grid_size - 1)
    cy = 0.5 * (grid_size - 1)
    cz = 0.5 * (grid_depth - 1)
    vol: SDFVolume = []
    for z in range(grid_depth):
        plane: list[list[float]] = []
        for y in range(grid_size):
            row = []
            for x in range(grid_size):
                row.append(_sphere_sdf(float(x), float(y), float(z), cx, cy, cz, radius))
            plane.append(row)
        vol.append(plane)
    return vol


def _default_recipe(run_index: int) -> dict[str, float]:
    return {
        "pressure": 20.0 + 2.0 * run_index,
        "rf_power": 100.0 + 5.0 * run_index,
        "gas_ratio": 0.40 + 0.03 * run_index,
    }


def _generate_run(
    run_id: str,
    num_steps: int,
    grid_size: int,
    dt: float,
    initial_radius: float,
    shrink_rate: float,
    recipe: dict[str, float],
    dimension: int = 2,
    grid_depth: int = 1,
) -> SyntheticSDFRun:
    phi_t: list[object] = []
    for step in range(num_steps):
        radius = max(0.1, initial_radius - shrink_rate * step * dt)
        if int(dimension) >= 3:
            phi_t.append(_build_phi_volume(grid_size=grid_size, grid_depth=max(2, int(grid_depth)), radius=radius))
        else:
            phi_t.append(_build_phi_frame(grid_size=grid_size, radius=radius))
    return SyntheticSDFRun(run_id=run_id, dt=dt, recipe=recipe, phi_t=phi_t)


def generate_synthetic_sdf_dataset(
    num_runs: int = 1,
    num_steps: int = 6,
    grid_size: int = 24,
    dt: float = 0.1,
    dimension: int = 2,
    grid_depth: int = 1,
) -> SyntheticSDFDataset:
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")
    if num_steps < 2:
        raise ValueError("num_steps must be >= 2")
    if grid_size < 4:
        raise ValueError("grid_size must be >= 4")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if int(dimension) not in {2, 3}:
        raise ValueError("dimension must be 2 or 3")
    if int(dimension) == 3 and int(grid_depth) < 2:
        raise ValueError("grid_depth must be >= 2 for 3D generation")

    runs: list[SyntheticSDFRun] = []
    for run_index in range(num_runs):
        run_id = f"synthetic_{run_index:03d}"
        scale_ref = grid_size if int(dimension) == 2 else min(grid_size, max(2, int(grid_depth)))
        initial_radius = 0.35 * float(scale_ref) + 0.3 * run_index
        shrink_rate = 0.55 + 0.05 * run_index
        runs.append(
            _generate_run(
                run_id=run_id,
                num_steps=num_steps,
                grid_size=grid_size,
                dt=dt,
                initial_radius=initial_radius,
                shrink_rate=shrink_rate,
                recipe=_default_recipe(run_index),
                dimension=int(dimension),
                grid_depth=int(grid_depth),
            )
        )
    return SyntheticSDFDataset(runs=runs)


def write_synthetic_example(
    output_dir: str | Path = "runs",
    num_runs: int = 1,
    num_steps: int = 6,
    grid_size: int = 24,
    dt: float = 0.1,
    dimension: int = 2,
    grid_depth: int = 1,
) -> Path:
    dataset = generate_synthetic_sdf_dataset(
        num_runs=num_runs,
        num_steps=num_steps,
        grid_size=grid_size,
        dt=dt,
        dimension=dimension,
        grid_depth=grid_depth,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = out_dir / f"synthetic_sdf_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(dataset.to_dict(), fp, indent=2)
    return output_path


def generate_proxy_mc_privileged_features(dataset: SyntheticSDFDataset) -> dict[str, dict[int, dict[str, float]]]:
    """Build proxy MC privileged features keyed by run_id/step_index."""
    from wafer_surrogate.data.mc_logs import generate_proxy_privileged_lookup

    lookup = generate_proxy_privileged_lookup(dataset)
    out: dict[str, dict[int, dict[str, float]]] = {}
    for (run_id, step_index), payload in lookup.items():
        out.setdefault(str(run_id), {})[int(step_index)] = {str(key): float(value) for key, value in payload.items()}
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wafer-surrogate-synthetic",
        description="Generate small synthetic SDF time-series datasets.",
    )
    parser.add_argument("--output-dir", default="runs", help="Directory for generated file.")
    parser.add_argument("--runs", type=int, default=1, help="Number of synthetic runs.")
    parser.add_argument("--steps", type=int, default=6, help="Time steps per run.")
    parser.add_argument("--grid-size", type=int, default=24, help="Square grid side length.")
    parser.add_argument("--dimension", type=int, default=2, choices=[2, 3], help="Generate 2D or 3D SDF.")
    parser.add_argument("--grid-depth", type=int, default=16, help="Depth size when --dimension 3.")
    parser.add_argument("--dt", type=float, default=0.1, help="Time delta between steps.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    output_path = write_synthetic_example(
        output_dir=args.output_dir,
        num_runs=args.runs,
        num_steps=args.steps,
        grid_size=args.grid_size,
        dt=args.dt,
        dimension=args.dimension,
        grid_depth=args.grid_depth,
    )
    print(f"synthetic dataset written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
