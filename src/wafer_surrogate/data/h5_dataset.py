from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wafer_surrogate.data.io import (
    NarrowBandDataset,
    NarrowBandRun,
    NarrowBandStep,
    read_hdf5_dataset,
    read_zarr_dataset,
)


@dataclass(frozen=True)
class SplitPolicy:
    seed: int = 0
    train_ratio: float = 0.8
    strict_split: bool = False

    def split(self, run_ids: list[str]) -> tuple[set[str], set[str]]:
        if not run_ids:
            return set(), set()
        uniq = sorted(set(str(v) for v in run_ids))
        if len(uniq) < 2:
            if bool(self.strict_split):
                return set(uniq), set()
            return set(uniq), set()
        rng = random.Random(int(self.seed))
        rng.shuffle(uniq)
        n_train = max(1, int(len(uniq) * float(self.train_ratio)))
        if n_train >= len(uniq):
            n_train = len(uniq) - 1
        train_ids = set(uniq[:n_train])
        valid_ids = set(uniq[n_train:])
        return train_ids, valid_ids


@dataclass(frozen=True)
class PointSampler:
    max_points: int = 0
    seed: int = 0
    patch_size: int = 0
    patches_per_step: int = 1

    def sample_indices(self, coords: list[list[int]]) -> list[int]:
        n = len(coords)
        if n < 1:
            return []
        idxs = list(range(n))
        rng = random.Random(int(self.seed))

        if int(self.patch_size) > 0 and n > 1:
            half = max(1, int(self.patch_size) // 2)
            patch_count = max(1, int(self.patches_per_step))
            chosen: set[int] = set()
            centers = [idxs[rng.randrange(len(idxs))] for _ in range(patch_count)]
            for center_idx in centers:
                cx, cy, cz = [int(v) for v in coords[center_idx]]
                for point_idx, coord in enumerate(coords):
                    x, y, z = [int(v) for v in coord]
                    if abs(x - cx) <= half and abs(y - cy) <= half and abs(z - cz) <= half:
                        chosen.add(int(point_idx))
            if chosen:
                idxs = sorted(chosen)

        if int(self.max_points) > 0 and len(idxs) > int(self.max_points):
            rng.shuffle(idxs)
            idxs = sorted(idxs[: max(1, int(self.max_points))])
        return idxs


class NarrowBandDatasetReader:
    """Unified narrow-band reader + sampler used by sparse train backends."""

    def __init__(self, dataset: NarrowBandDataset) -> None:
        self.dataset = dataset

    @classmethod
    def from_path(cls, path: str | Path) -> "NarrowBandDatasetReader":
        return cls(load_narrow_band_dataset(path))

    def split_runs(
        self,
        *,
        seed: int,
        train_ratio: float = 0.8,
        strict_split: bool = False,
    ) -> tuple[set[str], set[str]]:
        return SplitPolicy(seed=seed, train_ratio=train_ratio, strict_split=strict_split).split(
            [str(run.run_id) for run in self.dataset.runs]
        )

    def iter_step_records(
        self,
        *,
        include_priv: bool,
        sampler: PointSampler | None = None,
        run_filter: set[str] | None = None,
        latent_dim: int = 0,
        run_balance: bool = False,
    ) -> Iterator[dict[str, Any]]:
        runs = list(self.dataset.runs)
        if run_balance and len(runs) > 1:
            seed_value = int(sampler.seed) if sampler is not None else 0
            random.Random(seed_value).shuffle(runs)
        for run in runs:
            run_id = str(run.run_id)
            if run_filter is not None and run_id not in run_filter:
                continue
            cond_vec = [float(v) for v in run.recipe]
            if int(latent_dim) > 0:
                cond_vec.extend([0.0 for _ in range(int(latent_dim))])
            for step_idx, step in enumerate(run.steps):
                n = min(len(step.coords), len(step.feat), len(step.vn_target))
                if n < 1:
                    continue
                coords_src = [[int(v) for v in step.coords[idx]] for idx in range(n)]
                idxs = list(range(n)) if sampler is None else sampler.sample_indices(coords_src)
                if not idxs:
                    continue

                coords: list[list[int]] = []
                student_feat: list[list[float]] = []
                teacher_feat: list[list[float]] = []
                targets: list[float] = []
                for point_idx in idxs:
                    coord_xyz = [int(v) for v in step.coords[point_idx]]
                    base_feat = [float(v) for v in step.feat[point_idx]]
                    feat_row = list(base_feat)
                    feat_row.extend([float(coord_xyz[0]), float(coord_xyz[1]), float(coord_xyz[2])])
                    feat_row.append(float(step_idx))
                    student_row = list(feat_row)
                    teacher_row = list(feat_row)
                    if include_priv and step.priv is not None and point_idx < len(step.priv):
                        teacher_row.extend(float(v) for v in step.priv[point_idx])

                    coords.append(coord_xyz)
                    student_feat.append(student_row)
                    teacher_feat.append(teacher_row)
                    targets.append(float(step.vn_target[point_idx][0]))

                if not targets:
                    continue
                yield {
                    "run_id": run_id,
                    "step_index": int(step_idx),
                    "coords": coords,
                    "student_feat": student_feat,
                    "teacher_feat": teacher_feat,
                    "targets": targets,
                    "condition": cond_vec,
                }


def _load_json_narrow_band(path: Path) -> NarrowBandDataset:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"narrow-band json must be a mapping: {path}")
    runs_payload = payload.get("runs")
    if not isinstance(runs_payload, list) or not runs_payload:
        raise ValueError(f"narrow-band json missing non-empty runs: {path}")
    runs: list[NarrowBandRun] = []
    for run_idx, run_item in enumerate(runs_payload):
        if not isinstance(run_item, dict):
            raise ValueError(f"runs[{run_idx}] must be mapping")
        steps_payload = run_item.get("steps")
        if not isinstance(steps_payload, list) or not steps_payload:
            raise ValueError(f"runs[{run_idx}].steps must be non-empty list")
        steps: list[NarrowBandStep] = []
        for step_idx, step_item in enumerate(steps_payload):
            if not isinstance(step_item, dict):
                raise ValueError(f"runs[{run_idx}].steps[{step_idx}] must be mapping")
            priv_raw = step_item.get("priv")
            priv = None
            if isinstance(priv_raw, list):
                priv = [[float(cell) for cell in row] for row in priv_raw]
            steps.append(
                NarrowBandStep(
                    coords=[[int(cell) for cell in row] for row in step_item.get("coords", [])],
                    feat=[[float(cell) for cell in row] for row in step_item.get("feat", [])],
                    vn_target=[[float(cell) for cell in row] for row in step_item.get("vn_target", [])],
                    priv=priv,
                )
            )
        runs.append(
            NarrowBandRun(
                run_id=str(run_item.get("run_id", f"run_{run_idx:03d}")),
                recipe=[float(v) for v in run_item.get("recipe", [])],
                dt=float(run_item.get("dt", 0.1)),
                steps=steps,
            )
        )
    return NarrowBandDataset(runs=runs)


def load_narrow_band_dataset(path: str | Path) -> NarrowBandDataset:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"narrow-band path does not exist: {p}")
    suffix = p.suffix.lower()
    if suffix == ".h5":
        return read_hdf5_dataset(p)
    if suffix == ".zarr" or p.name.endswith(".zarr"):
        return read_zarr_dataset(p)
    if suffix == ".json":
        return _load_json_narrow_band(p)
    raise ValueError(f"unsupported narrow-band path format: {p}")


def split_runs(
    run_ids: list[str],
    *,
    seed: int,
    train_ratio: float = 0.8,
    strict_split: bool = False,
) -> tuple[set[str], set[str]]:
    return SplitPolicy(seed=seed, train_ratio=train_ratio, strict_split=strict_split).split(run_ids)
