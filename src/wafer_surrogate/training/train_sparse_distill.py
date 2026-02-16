from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import random
from statistics import fmean
from typing import Any

from wafer_surrogate.data.h5_dataset import NarrowBandDatasetReader, PointSampler
from wafer_surrogate.data.io import NarrowBandDataset
from wafer_surrogate.models.sparse_unet_film import (
    FeatureContract,
    OptionalSparseDependencyUnavailable,
    SparseTensorVnModel,
    SparseUNetFiLM,
    _require_sparse_dependencies,
)


@dataclass(frozen=True)
class SparseDistillConfig:
    teacher_epochs: int = 40
    student_epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    alpha: float = 0.5
    beta: float = 0.2
    gamma: float = 0.1
    batch_size_steps: int = 8
    max_points: int = 20000
    seed: int = 0
    device: str = "cpu"
    recipe_keys: list[str] = field(default_factory=list)
    band_width: float = 0.5
    min_grad_norm: float = 1e-6
    grad_clip_norm: float = 5.0
    early_stopping_patience: int = 0
    lr_scheduler: str = "none"
    latent_dim: int = 0
    rollout_loss_enabled: bool = False
    rollout_k: int = 3
    rollout_weight: float = 0.1
    temporal_step_weight: str = "uniform"
    step_sampling_policy: str = "uniform"
    strict_split: bool = False
    patch_size: int = 0
    patches_per_step: int = 1
    sparse_model_profile: str = "small"
    hidden_channels: int = 0
    num_blocks: int = 0
    dropout: float = 0.0
    residual: bool = True


@dataclass
class SparseDistillResult:
    model: SparseTensorVnModel
    metrics: dict[str, float]
    distill_metrics: dict[str, float]
    metric_rows: list[dict[str, Any]]
    comparisons: list[dict[str, Any]]
    split_info: dict[str, Any]
    feature_contract: FeatureContract
    condition_scaler: dict[str, Any]
    temporal_diagnostics: dict[str, Any]
    checkpoint_path: Path
    train_true: list[float]
    train_pred: list[float]
    valid_true: list[float]
    valid_pred: list[float]


def _mae(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        return 0.0
    return fmean(abs(float(t) - float(p)) for t, p in zip(y_true, y_pred))


def _rmse(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        return 0.0
    return fmean((float(t) - float(p)) ** 2 for t, p in zip(y_true, y_pred)) ** 0.5


def _split_run_ids(
    dataset: NarrowBandDataset,
    *,
    seed: int,
    train_ratio: float = 0.8,
    strict_split: bool = False,
) -> tuple[set[str], set[str]]:
    reader = NarrowBandDatasetReader(dataset)
    return reader.split_runs(
        seed=int(seed),
        train_ratio=float(train_ratio),
        strict_split=bool(strict_split),
    )


def _build_condition_scaler(*, dataset: NarrowBandDataset, train_run_ids: set[str], latent_dim: int) -> dict[str, Any]:
    vectors: list[list[float]] = []
    for run in dataset.runs:
        if str(run.run_id) not in train_run_ids:
            continue
        vec = [float(v) for v in run.recipe]
        if int(latent_dim) > 0:
            vec.extend([0.0 for _ in range(int(latent_dim))])
        vectors.append(vec)
    if not vectors:
        return {"schema_version": 1, "enabled": False, "mean": [], "std": []}
    dim = len(vectors[0])
    means: list[float] = []
    stds: list[float] = []
    for col in range(dim):
        col_vals = [float(row[col]) for row in vectors]
        mu = float(fmean(col_vals))
        means.append(mu)
        variance = float(fmean((value - mu) ** 2 for value in col_vals))
        sigma = variance ** 0.5
        if sigma < 1e-12:
            sigma = 1.0
        stds.append(float(sigma))
    return {"schema_version": 1, "enabled": True, "mean": means, "std": stds}


def _apply_condition_scaler(values: list[float], scaler: dict[str, Any]) -> list[float]:
    if not bool(scaler.get("enabled", False)):
        return list(values)
    mean = scaler.get("mean")
    std = scaler.get("std")
    if not isinstance(mean, list) or not isinstance(std, list):
        return list(values)
    out: list[float] = []
    for idx, value in enumerate(values):
        mu = float(mean[idx]) if idx < len(mean) else 0.0
        sigma = float(std[idx]) if idx < len(std) else 1.0
        if abs(sigma) < 1e-12:
            sigma = 1.0
        out.append((float(value) - mu) / sigma)
    return out


def _temporal_weight(step_index: int, policy: str) -> float:
    mode = str(policy).strip().lower()
    if mode == "early_decay":
        if step_index <= 2:
            return 2.0
        return max(0.5, 1.0 / (1.0 + (0.25 * float(step_index - 2))))
    return 1.0


def _iter_batches(
    records_iter: Iterator[dict[str, Any]],
    *,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    step = max(1, int(batch_size))
    batch: list[dict[str, Any]] = []
    for record in records_iter:
        batch.append(record)
        if len(batch) >= step:
            yield batch
            batch = []
    if batch:
        yield batch


def _collate_for_sparse(batch: list[dict[str, Any]], *, use_teacher_feat: bool, device: str) -> dict[str, Any]:
    torch_mod = _require_sparse_dependencies()[0]
    coords_all: list[list[int]] = []
    feat_all: list[list[float]] = []
    target_all: list[float] = []
    cond_all: list[list[float]] = []
    point_counts: list[int] = []
    record_meta: list[dict[str, Any]] = []

    for b_idx, record in enumerate(batch):
        cond_all.append([float(v) for v in record["condition"]])
        coords = record["coords"]
        feats = record["teacher_feat"] if use_teacher_feat else record["student_feat"]
        targets = record["targets"]
        point_counts.append(len(targets))
        record_meta.append({"run_id": str(record["run_id"]), "step_index": int(record["step_index"])})

        for coord, feat, target in zip(coords, feats, targets):
            coords_all.append([int(b_idx), int(coord[0]), int(coord[1]), int(coord[2])])
            feat_all.append([float(v) for v in feat])
            target_all.append(float(target))

    coords_t = torch_mod.as_tensor(coords_all, dtype=torch_mod.int32, device=device)
    feat_t = torch_mod.as_tensor(feat_all, dtype=torch_mod.float32, device=device)
    target_t = torch_mod.as_tensor(target_all, dtype=torch_mod.float32, device=device)
    cond_t = torch_mod.as_tensor(cond_all, dtype=torch_mod.float32, device=device)
    return {
        "coords": coords_t,
        "feat": feat_t,
        "target": target_t,
        "cond": cond_t,
        "point_counts": point_counts,
        "record_meta": record_meta,
    }


def _point_temporal_weights(
    *,
    packed: dict[str, Any],
    policy: str,
) -> Any:
    torch_mod = _require_sparse_dependencies()[0]
    weights: list[float] = []
    for point_count, meta in zip(packed["point_counts"], packed["record_meta"]):
        step_index = int(meta.get("step_index", 0))
        value = float(_temporal_weight(step_index, policy))
        weights.extend([value for _ in range(max(0, int(point_count)))])
    if not weights:
        return torch_mod.ones((0,), dtype=torch_mod.float32, device=packed["target"].device)
    return torch_mod.as_tensor(weights, dtype=torch_mod.float32, device=packed["target"].device)


def _weighted_mse(pred: Any, target: Any, weights: Any) -> Any:
    diff = (pred - target) ** 2
    if int(diff.numel()) == 0:
        return diff.new_tensor(0.0)
    if int(weights.numel()) != int(diff.numel()):
        return diff.mean()
    denom = weights.sum().clamp_min(1e-12)
    return (diff * weights).sum() / denom


def _weighted_feature_mse(pred: Any, target: Any, weights: Any) -> Any:
    if int(pred.numel()) == 0 or int(target.numel()) == 0:
        return pred.new_tensor(0.0)
    per_point = ((pred - target) ** 2).mean(dim=1)
    if int(weights.numel()) != int(per_point.numel()):
        return per_point.mean()
    denom = weights.sum().clamp_min(1e-12)
    return (per_point * weights).sum() / denom


def _evaluate_model(
    model: Any,
    records_factory: Callable[[], Iterator[dict[str, Any]]],
    *,
    use_teacher_feat: bool,
    device: str,
) -> tuple[list[float], list[float]]:
    torch_mod, _, me_mod = _require_sparse_dependencies()
    preds: list[float] = []
    trues: list[float] = []

    model.eval()
    with torch_mod.no_grad():
        for batch in _iter_batches(records_factory(), batch_size=8):
            packed = _collate_for_sparse(batch, use_teacher_feat=use_teacher_feat, device=device)
            st = me_mod.SparseTensor(features=packed["feat"], coordinates=packed["coords"])
            out = model(st, packed["cond"])  # sparse tensor
            y_pred = out.F.reshape(-1).detach().cpu().tolist()
            y_true = packed["target"].reshape(-1).detach().cpu().tolist()
            preds.extend(float(v) for v in y_pred)
            trues.extend(float(v) for v in y_true)
    return trues, preds


def _temporal_vn_diagnostics(
    model: Any,
    records_factory: Callable[[], Iterator[dict[str, Any]]],
    *,
    use_teacher_feat: bool,
    device: str,
) -> dict[str, Any]:
    torch_mod, _, me_mod = _require_sparse_dependencies()
    by_run_step_true: dict[tuple[str, int], list[float]] = {}
    by_run_step_pred: dict[tuple[str, int], list[float]] = {}
    model.eval()
    with torch_mod.no_grad():
        for batch in _iter_batches(records_factory(), batch_size=8):
            packed = _collate_for_sparse(batch, use_teacher_feat=use_teacher_feat, device=device)
            st = me_mod.SparseTensor(features=packed["feat"], coordinates=packed["coords"])
            out = model(st, packed["cond"]).F.reshape(-1)
            target = packed["target"].reshape(-1)
            offset = 0
            for point_count, meta in zip(packed["point_counts"], packed["record_meta"]):
                n = max(0, int(point_count))
                if n == 0:
                    continue
                key = (str(meta["run_id"]), int(meta["step_index"]))
                pred_chunk = out[offset : offset + n]
                true_chunk = target[offset : offset + n]
                by_run_step_pred.setdefault(key, []).append(float(pred_chunk.mean().detach().cpu().item()))
                by_run_step_true.setdefault(key, []).append(float(true_chunk.mean().detach().cpu().item()))
                offset += n

    per_step_mae: dict[str, float] = {}
    early_vals: list[float] = []
    late_vals: list[float] = []
    sign_hits = 0
    sign_total = 0
    step_pairs: list[tuple[int, float, float]] = []
    for key in sorted(by_run_step_true.keys()):
        run_id, step_index = key
        true_val = float(fmean(by_run_step_true.get((run_id, step_index), [0.0])))
        pred_val = float(fmean(by_run_step_pred.get((run_id, step_index), [0.0])))
        err = abs(pred_val - true_val)
        step_pairs.append((int(step_index), pred_val, true_val))
        per_step_mae[str(step_index)] = float(err)
        if step_index <= 2:
            early_vals.append(float(err))
        else:
            late_vals.append(float(err))
        if abs(true_val) > 1e-12:
            sign_total += 1
            if (pred_val >= 0.0) == (true_val >= 0.0):
                sign_hits += 1

    return {
        "vn_mae_per_step": per_step_mae,
        "early_window_vn_mae": float(fmean(early_vals)) if early_vals else 0.0,
        "late_window_vn_mae": float(fmean(late_vals)) if late_vals else 0.0,
        "vn_sign_agreement": (float(sign_hits) / float(sign_total)) if sign_total else 0.0,
        "num_step_records": int(len(step_pairs)),
    }


def _resolve_sparse_arch(config: SparseDistillConfig) -> dict[str, Any]:
    presets = {
        "small": {"hidden_channels": 32, "num_blocks": 2, "dropout": 0.0, "residual": True},
        "base": {"hidden_channels": 64, "num_blocks": 3, "dropout": 0.1, "residual": True},
        "large": {"hidden_channels": 96, "num_blocks": 4, "dropout": 0.15, "residual": True},
    }
    profile = str(config.sparse_model_profile).strip().lower() or "small"
    base = dict(presets.get(profile, presets["small"]))
    if int(config.hidden_channels) > 0:
        base["hidden_channels"] = int(config.hidden_channels)
    if int(config.num_blocks) > 0:
        base["num_blocks"] = int(config.num_blocks)
    if float(config.dropout) > 0.0:
        base["dropout"] = float(config.dropout)
    base["residual"] = bool(config.residual)
    base["profile"] = profile if profile in presets else "custom"
    return base


def train_sparse_distill(
    *,
    dataset: NarrowBandDataset,
    output_dir: Path,
    config: SparseDistillConfig,
) -> SparseDistillResult:
    torch_mod, _, me_mod = _require_sparse_dependencies()

    reader = NarrowBandDatasetReader(dataset)
    train_run_ids, valid_run_ids = _split_run_ids(
        dataset,
        seed=int(config.seed),
        train_ratio=0.8,
        strict_split=bool(config.strict_split),
    )
    if bool(config.strict_split) and len(valid_run_ids) < 1:
        raise ValueError("strict_split=true requires num_valid_runs>0 (insufficient_unique_runs)")
    include_priv = any(step.priv is not None for run in dataset.runs for step in run.steps)
    condition_scaler = _build_condition_scaler(
        dataset=dataset,
        train_run_ids=train_run_ids,
        latent_dim=max(0, int(config.latent_dim)),
    )
    sampling_policy = str(config.step_sampling_policy).strip().lower() or "uniform"
    if sampling_policy not in {"uniform", "early_bias"}:
        raise ValueError(f"unsupported step_sampling_policy: {config.step_sampling_policy}")

    def _records(run_filter: set[str], *, seed: int) -> Iterator[dict[str, Any]]:
        sampler = PointSampler(
            max_points=max(0, int(config.max_points)),
            seed=int(seed),
            patch_size=max(0, int(config.patch_size)),
            patches_per_step=max(1, int(config.patches_per_step)),
        )
        rng = random.Random(int(seed) + 97)
        for record in reader.iter_step_records(
            include_priv=bool(include_priv),
            sampler=sampler,
            run_filter=run_filter,
            latent_dim=max(0, int(config.latent_dim)),
            run_balance=True,
        ):
            step_index = int(record["step_index"])
            repeats = 1
            if sampling_policy == "early_bias":
                if step_index <= 2:
                    repeats = 2
                elif step_index >= 6 and rng.random() < 0.3:
                    repeats = 0
            if repeats < 1:
                continue
            cond_scaled = _apply_condition_scaler(
                [float(v) for v in record["condition"]],
                condition_scaler,
            )
            out = dict(record)
            out["condition"] = cond_scaled
            for _ in range(repeats):
                yield out

    run_dt: dict[str, float] = {str(run.run_id): float(run.dt) for run in dataset.runs}
    step_phi_mean: dict[tuple[str, int], float] = {}
    step_grad_mean: dict[tuple[str, int], float] = {}
    step_vn_mean: dict[tuple[str, int], float] = {}
    train_step_counts: dict[int, int] = {}
    valid_step_counts: dict[int, int] = {}
    train_record_count = 0
    valid_record_count = 0
    first_train_record: dict[str, Any] | None = None
    for record in _records(train_run_ids, seed=int(config.seed)):
        train_record_count += 1
        step_idx_count = int(record["step_index"])
        train_step_counts[step_idx_count] = int(train_step_counts.get(step_idx_count, 0) + 1)
        if first_train_record is None:
            first_train_record = record
        run_id = str(record["run_id"])
        step_index = int(record["step_index"])
        feats = record["student_feat"]
        targets = record["targets"]
        if feats:
            phi_values = [float(row[0]) for row in feats]
            grad_values = [float(row[1]) for row in feats if len(row) > 1]
            step_phi_mean[(run_id, step_index)] = float(fmean(phi_values))
            step_grad_mean[(run_id, step_index)] = float(fmean(grad_values)) if grad_values else 1.0
        else:
            step_phi_mean[(run_id, step_index)] = 0.0
            step_grad_mean[(run_id, step_index)] = 1.0
        step_vn_mean[(run_id, step_index)] = float(fmean(float(v) for v in targets)) if targets else 0.0

    for record in _records(valid_run_ids, seed=int(config.seed) + 17):
        valid_record_count += 1
        step_idx_count = int(record["step_index"])
        valid_step_counts[step_idx_count] = int(valid_step_counts.get(step_idx_count, 0) + 1)

    if first_train_record is None:
        raise ValueError("sparse distill training produced no train records")

    cond_dim = max(1, len(first_train_record["condition"]))
    student_feat_dim = max(1, len(first_train_record["student_feat"][0]))
    teacher_feat_dim = max(1, len(first_train_record["teacher_feat"][0]))

    base_recipe_keys = [str(v) for v in config.recipe_keys] if config.recipe_keys else [f"recipe_{i}" for i in range(max(0, cond_dim - max(0, int(config.latent_dim))))]
    latent_keys = [f"z_{idx:02d}" for idx in range(max(0, int(config.latent_dim)))]
    recipe_keys = list(base_recipe_keys) + latent_keys
    feature_names: list[str] = [
        "phi",
        "grad_norm",
        "curvature_proxy",
        "band_distance",
        "coord_x",
        "coord_y",
        "coord_z",
        "step_index",
    ]
    feature_names = feature_names[:student_feat_dim]
    while len(feature_names) < student_feat_dim:
        feature_names.append(f"feat_{len(feature_names)}")

    contract = FeatureContract(
        recipe_keys=recipe_keys,
        feature_names=feature_names,
        cond_dim=cond_dim,
        feat_dim=student_feat_dim,
        band_width=float(config.band_width),
        min_grad_norm=float(config.min_grad_norm),
    )

    arch = _resolve_sparse_arch(config)
    teacher_model = SparseUNetFiLM(
        in_channels=teacher_feat_dim,
        cond_dim=cond_dim,
        hidden_channels=int(arch["hidden_channels"]),
        num_blocks=int(arch["num_blocks"]),
        dropout=float(arch["dropout"]),
        residual=bool(arch["residual"]),
        out_channels=1,
    ).to(config.device)
    student_model = SparseUNetFiLM(
        in_channels=student_feat_dim,
        cond_dim=cond_dim,
        hidden_channels=int(arch["hidden_channels"]),
        num_blocks=int(arch["num_blocks"]),
        dropout=float(arch["dropout"]),
        residual=bool(arch["residual"]),
        out_channels=1,
    ).to(config.device)

    teacher_opt = torch_mod.optim.Adam(teacher_model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    student_opt = torch_mod.optim.Adam(student_model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))

    scheduler_name = str(config.lr_scheduler).strip().lower() or "none"
    teacher_scheduler = None
    student_scheduler = None
    if scheduler_name == "cosine":
        teacher_scheduler = torch_mod.optim.lr_scheduler.CosineAnnealingLR(
            teacher_opt,
            T_max=max(1, int(config.teacher_epochs)),
        )
        student_scheduler = torch_mod.optim.lr_scheduler.CosineAnnealingLR(
            student_opt,
            T_max=max(1, int(config.student_epochs)),
        )
    elif scheduler_name != "none":
        raise ValueError(f"unsupported sparse_distill lr_scheduler: {config.lr_scheduler}")

    # Train teacher.
    teacher_model.train()
    for epoch in range(max(1, int(config.teacher_epochs))):
        train_iter = _records(train_run_ids, seed=int(config.seed) + epoch)
        for batch in _iter_batches(train_iter, batch_size=max(1, int(config.batch_size_steps))):
            packed = _collate_for_sparse(batch, use_teacher_feat=True, device=config.device)
            point_weights = _point_temporal_weights(
                packed=packed,
                policy=str(config.temporal_step_weight),
            )
            teacher_opt.zero_grad(set_to_none=True)
            st = me_mod.SparseTensor(features=packed["feat"], coordinates=packed["coords"])
            out = teacher_model(st, packed["cond"])  # sparse tensor
            loss = _weighted_mse(out.F.reshape(-1), packed["target"].reshape(-1), point_weights)
            loss.backward()
            if float(config.grad_clip_norm) > 0:
                torch_mod.nn.utils.clip_grad_norm_(teacher_model.parameters(), max_norm=float(config.grad_clip_norm))
            teacher_opt.step()
        if teacher_scheduler is not None:
            teacher_scheduler.step()

    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

    alpha = float(config.alpha)
    beta = float(config.beta)
    gamma = float(config.gamma)
    denom = alpha + beta + gamma
    if denom <= 0.0:
        alpha, beta, gamma = 1.0, 0.0, 0.0
        denom = 1.0
    alpha /= denom
    beta /= denom
    gamma /= denom

    # Train student with distillation.
    student_model.train()
    rollout_loss_total = 0.0
    rollout_loss_count = 0
    best_epoch = 0
    best_valid_mae = float("inf")
    best_state: dict[str, Any] | None = None
    no_improve_epochs = 0
    stopped_early = False
    for epoch in range(max(1, int(config.student_epochs))):
        train_iter = _records(train_run_ids, seed=int(config.seed) + 1000 + epoch)
        for batch in _iter_batches(train_iter, batch_size=max(1, int(config.batch_size_steps))):
            packed_teacher = _collate_for_sparse(batch, use_teacher_feat=True, device=config.device)
            packed_student = _collate_for_sparse(batch, use_teacher_feat=False, device=config.device)
            point_weights = _point_temporal_weights(
                packed=packed_student,
                policy=str(config.temporal_step_weight),
            )
            st_teacher = me_mod.SparseTensor(features=packed_teacher["feat"], coordinates=packed_teacher["coords"])
            st_student = me_mod.SparseTensor(features=packed_student["feat"], coordinates=packed_student["coords"])

            with torch_mod.no_grad():
                teacher_out, teacher_mid = teacher_model(st_teacher, packed_teacher["cond"], return_features=True)

            student_opt.zero_grad(set_to_none=True)
            student_out, student_mid = student_model(st_student, packed_student["cond"], return_features=True)

            target_loss = _weighted_mse(student_out.F.reshape(-1), packed_student["target"].reshape(-1), point_weights)
            teacher_loss = _weighted_mse(student_out.F.reshape(-1), teacher_out.F.reshape(-1), point_weights)
            feature_loss = _weighted_feature_mse(student_mid, teacher_mid, point_weights)
            rollout_loss = None
            if bool(config.rollout_loss_enabled):
                offset = 0
                pred_step_vn: dict[tuple[str, int], float] = {}
                for point_count, meta in zip(packed_student["point_counts"], packed_student["record_meta"]):
                    n = int(point_count)
                    if n <= 0:
                        continue
                    run_id = str(meta["run_id"])
                    step_index = int(meta["step_index"])
                    chunk = student_out.F.reshape(-1)[offset : offset + n]
                    pred_step_vn[(run_id, step_index)] = float(chunk.mean().detach().cpu().item())
                    offset += n

                rollout_err: list[Any] = []
                rollout_weight_sum = 0.0
                rollout_weighted_err = 0.0
                horizon = max(1, int(config.rollout_k))
                for meta in packed_student["record_meta"]:
                    run_id = str(meta["run_id"])
                    step_index = int(meta["step_index"])
                    if (run_id, step_index) not in step_phi_mean:
                        continue
                    dt = float(run_dt.get(run_id, 0.1))
                    phi_pred = float(step_phi_mean[(run_id, step_index)])
                    for h in range(horizon):
                        cur_step = step_index + h
                        next_step = cur_step + 1
                        true_next_phi = step_phi_mean.get((run_id, next_step))
                        if true_next_phi is None:
                            break
                        vn_value = pred_step_vn.get((run_id, cur_step), step_vn_mean.get((run_id, cur_step), 0.0))
                        grad_value = max(1e-6, abs(float(step_grad_mean.get((run_id, cur_step), 1.0))))
                        phi_pred = float(phi_pred) - (dt * float(vn_value) * grad_value)
                        sq_err = (phi_pred - float(true_next_phi)) ** 2
                        rollout_err.append(sq_err)
                        step_weight = _temporal_weight(cur_step, str(config.temporal_step_weight))
                        rollout_weighted_err += float(step_weight) * float(sq_err)
                        rollout_weight_sum += float(step_weight)
                if rollout_err:
                    if rollout_weight_sum > 0.0:
                        rollout_value = float(rollout_weighted_err / rollout_weight_sum)
                    else:
                        rollout_value = float(fmean(float(v) for v in rollout_err))
                    rollout_loss = student_out.F.new_tensor(rollout_value)
                    rollout_loss_total += float(rollout_loss.detach().cpu().item())
                    rollout_loss_count += 1
                else:
                    rollout_loss = student_out.F.new_tensor(0.0)

            loss = (alpha * target_loss) + (beta * teacher_loss) + (gamma * feature_loss)
            if rollout_loss is not None:
                loss = loss + (float(config.rollout_weight) * rollout_loss)
            loss.backward()
            if float(config.grad_clip_norm) > 0:
                torch_mod.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=float(config.grad_clip_norm))
            student_opt.step()
        if student_scheduler is not None:
            student_scheduler.step()

        valid_true_epoch, valid_pred_epoch = _evaluate_model(
            student_model,
            lambda: _records(
                valid_run_ids if valid_record_count > 0 else train_run_ids,
                seed=int(config.seed) + 40_000 + epoch,
            ),
            use_teacher_feat=False,
            device=config.device,
        )
        valid_mae_epoch = _mae(valid_true_epoch, valid_pred_epoch)
        if valid_mae_epoch + 1e-12 < best_valid_mae:
            best_valid_mae = float(valid_mae_epoch)
            best_epoch = int(epoch)
            no_improve_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in student_model.state_dict().items()
            }
        else:
            no_improve_epochs += 1
        patience = max(0, int(config.early_stopping_patience))
        if patience > 0 and no_improve_epochs >= patience:
            stopped_early = True
            break

    if best_state is not None:
        student_model.load_state_dict(best_state)

    train_true_t, train_pred_t = _evaluate_model(
        teacher_model,
        lambda: _records(train_run_ids, seed=int(config.seed) + 20_000),
        use_teacher_feat=True,
        device=config.device,
    )
    train_true_s, train_pred_s = _evaluate_model(
        student_model,
        lambda: _records(train_run_ids, seed=int(config.seed) + 20_001),
        use_teacher_feat=False,
        device=config.device,
    )

    if valid_record_count > 0:
        valid_true_s, valid_pred_s = _evaluate_model(
            student_model,
            lambda: _records(valid_run_ids, seed=int(config.seed) + 20_017),
            use_teacher_feat=False,
            device=config.device,
        )
    else:
        valid_true_s, valid_pred_s = train_true_s, train_pred_s

    train_temporal_diag = _temporal_vn_diagnostics(
        student_model,
        lambda: _records(train_run_ids, seed=int(config.seed) + 30_001),
        use_teacher_feat=False,
        device=config.device,
    )
    valid_temporal_diag = _temporal_vn_diagnostics(
        student_model,
        lambda: _records(
            valid_run_ids if valid_record_count > 0 else train_run_ids,
            seed=int(config.seed) + 30_017,
        ),
        use_teacher_feat=False,
        device=config.device,
    )

    teacher_mae = _mae(train_true_t, train_pred_t)
    student_mae = _mae(train_true_s, train_pred_s)
    student_rmse = _rmse(train_true_s, train_pred_s)
    valid_mae = _mae(valid_true_s, valid_pred_s)
    valid_rmse = _rmse(valid_true_s, valid_pred_s)
    distill_gap = float(student_mae - teacher_mae)

    model = SparseTensorVnModel(
        network=student_model,
        contract=contract,
        device=config.device,
        condition_scaler=condition_scaler,
        architecture={
            "sparse_model_profile": str(arch.get("profile", "small")),
            "hidden_channels": int(arch["hidden_channels"]),
            "num_blocks": int(arch["num_blocks"]),
            "dropout": float(arch["dropout"]),
            "residual": bool(arch["residual"]),
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model.save_checkpoint(output_dir / "sparse_student.pt")

    metrics = {
        "mae": float(student_mae),
        "rmse": float(student_rmse),
        "student_mae": float(student_mae),
        "teacher_mae": float(teacher_mae),
        "distill_gap": float(distill_gap),
        "vn_rmse": float(student_rmse),
        "cv_valid_mae": float(valid_mae),
        "cv_valid_rmse": float(valid_rmse),
        "target_mean": float(fmean(train_true_s)) if train_true_s else 0.0,
        "prediction_mean": float(fmean(train_pred_s)) if train_pred_s else 0.0,
        "rollout_loss": (float(rollout_loss_total) / float(rollout_loss_count)) if rollout_loss_count > 0 else 0.0,
        "rollout_short_window_error": (float(rollout_loss_total) / float(rollout_loss_count)) if rollout_loss_count > 0 else 0.0,
        "train_early_window_vn_mae": float(train_temporal_diag.get("early_window_vn_mae", 0.0)),
        "valid_early_window_vn_mae": float(valid_temporal_diag.get("early_window_vn_mae", 0.0)),
        "valid_vn_sign_agreement": float(valid_temporal_diag.get("vn_sign_agreement", 0.0)),
        "best_epoch": float(best_epoch),
        "best_valid_mae": float(best_valid_mae if best_valid_mae < float("inf") else valid_mae),
        "stopped_early": 1.0 if stopped_early else 0.0,
    }

    distill_metrics = {
        "teacher_mae": float(teacher_mae),
        "student_mae": float(student_mae),
        "distill_gap": float(distill_gap),
        "vn_rmse": float(student_rmse),
        "num_samples": float(len(train_true_s)),
        "cv_valid_mae": float(valid_mae),
        "cv_valid_rmse": float(valid_rmse),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "rollout_loss_enabled": bool(config.rollout_loss_enabled),
        "rollout_k": int(config.rollout_k),
        "rollout_weight": float(config.rollout_weight),
        "rollout_loss": (float(rollout_loss_total) / float(rollout_loss_count)) if rollout_loss_count > 0 else 0.0,
        "rollout_short_window_error": (float(rollout_loss_total) / float(rollout_loss_count)) if rollout_loss_count > 0 else 0.0,
        "temporal_step_weight": str(config.temporal_step_weight),
        "step_sampling_policy": sampling_policy,
        "train_temporal_diagnostics": dict(train_temporal_diag),
        "valid_temporal_diagnostics": dict(valid_temporal_diag),
        "best_epoch": int(best_epoch),
        "best_valid_mae": float(best_valid_mae if best_valid_mae < float("inf") else valid_mae),
        "stopped_early": bool(stopped_early),
        "lr_scheduler": scheduler_name,
        "step_sampling_distribution_train": {str(k): int(v) for k, v in sorted(train_step_counts.items())},
        "step_sampling_distribution_valid": {str(k): int(v) for k, v in sorted(valid_step_counts.items())},
    }

    metric_rows = [
        {"model": "sparse_vn_teacher", "variant_id": "teacher", "split": "train", "metric": "mae", "value": float(teacher_mae), "is_best": 0},
        {"model": "sparse_vn_student", "variant_id": "student", "split": "train", "metric": "mae", "value": float(student_mae), "is_best": 1},
        {"model": "sparse_vn_student", "variant_id": "student", "split": "train", "metric": "vn_rmse", "value": float(student_rmse), "is_best": 1},
        {"model": "sparse_vn_student", "variant_id": "student", "split": "cv", "metric": "valid_mae", "value": float(valid_mae), "is_best": 1},
        {"model": "sparse_vn_student", "variant_id": "student", "split": "cv", "metric": "valid_rmse", "value": float(valid_rmse), "is_best": 1},
    ]

    comparisons = [
        {
            "variant_id": "teacher",
            "model": "sparse_vn_teacher",
            "split": "train",
            "status": "ok",
            "error": "",
            "mae": float(teacher_mae),
            "rmse": float(_rmse(train_true_t, train_pred_t)),
            "target_mean": float(fmean(train_true_t)) if train_true_t else 0.0,
            "prediction_mean": float(fmean(train_pred_t)) if train_pred_t else 0.0,
            "cv_valid_mae": None,
            "cv_valid_rmse": None,
            "rank": 2,
            "is_best": 0,
        },
        {
            "variant_id": "student",
            "model": "sparse_vn_student",
            "split": "train",
            "status": "ok",
            "error": "",
            "mae": float(student_mae),
            "rmse": float(student_rmse),
            "target_mean": float(fmean(train_true_s)) if train_true_s else 0.0,
            "prediction_mean": float(fmean(train_pred_s)) if train_pred_s else 0.0,
            "cv_valid_mae": float(valid_mae),
            "cv_valid_rmse": float(valid_rmse),
            "rank": 1,
            "is_best": 1,
        },
    ]

    split_info = {
        "schema_version": 2,
        "mode": "sparse_distill_me",
        "seed": int(config.seed),
        "num_train_runs": len(train_run_ids),
        "num_valid_runs": len(valid_run_ids),
        "num_train_records": int(train_record_count),
        "num_valid_records": int(valid_record_count),
        "leak_checked": len(valid_run_ids) > 0,
        "strict_split_enforced": bool(config.strict_split and len(valid_run_ids) > 0),
        "reason": "" if len(valid_run_ids) > 0 else "insufficient_unique_runs",
        "patch_size": int(config.patch_size),
        "patches_per_step": int(config.patches_per_step),
        "loader_mode": "streaming",
        "step_sampling_policy": sampling_policy,
    }

    return SparseDistillResult(
        model=model,
        metrics=metrics,
        distill_metrics=distill_metrics,
        metric_rows=metric_rows,
        comparisons=comparisons,
        split_info=split_info,
        feature_contract=contract,
        condition_scaler=condition_scaler,
        temporal_diagnostics={
            "train": dict(train_temporal_diag),
            "valid": dict(valid_temporal_diag),
        },
        checkpoint_path=checkpoint_path,
        train_true=[float(v) for v in train_true_s],
        train_pred=[float(v) for v in train_pred_s],
        valid_true=[float(v) for v in valid_true_s],
        valid_pred=[float(v) for v in valid_pred_s],
    )
