from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from statistics import fmean

from wafer_surrogate.data.synthetic import SyntheticSDFRun
from wafer_surrogate.observation import (
    BaselineSdfObservationModel,
    compute_observation_feature_metrics,
)


def _frame_mean(frame: Sequence[Sequence[float]]) -> float:
    values = _flatten_values(frame)
    total = sum(values)
    count = len(values)
    if count == 0:
        raise ValueError("frame is empty")
    return total / count


def _flatten_values(frame: object) -> list[float]:
    if hasattr(frame, "tolist"):
        frame = frame.tolist()
    if isinstance(frame, Sequence) and not isinstance(frame, (str, bytes, bytearray)):
        out: list[float] = []
        for item in frame:
            out.extend(_flatten_values(item))
        return out
    return [float(frame)]  # type: ignore[arg-type]


def _profile_features(frame: object) -> dict[str, float]:
    if not frame:
        raise ValueError("frame is empty")
    if isinstance(frame, Sequence) and frame and isinstance(frame[0], Sequence) and frame[0] and isinstance(frame[0][0], Sequence):
        # 3D volume: aggregate profile metrics over representative depth slices.
        depth = len(frame)
        anchors = sorted({0, depth // 4, depth // 2, (3 * depth) // 4, max(depth - 1, 0)})
        acc = {
            "row_neg_fraction": 0.0,
            "row_min": 0.0,
            "col_neg_fraction": 0.0,
            "col_min": 0.0,
        }
        count = 0
        for z in anchors:
            plane = frame[z]  # type: ignore[index]
            sub = _profile_features(plane)
            for key in acc:
                acc[key] += float(sub[key])
            count += 1
        denom = float(max(1, count))
        return {key: value / denom for key, value in acc.items()}

    width = len(frame[0])  # type: ignore[index]
    if width == 0:
        raise ValueError("frame row is empty")
    if any(len(row) != width for row in frame):  # type: ignore[arg-type]
        raise ValueError("frame must be rectangular")

    mid_y = len(frame) // 2  # type: ignore[arg-type]
    mid_x = width // 2
    center_row = [float(value) for value in frame[mid_y]]  # type: ignore[index]
    center_col = [float(row[mid_x]) for row in frame]  # type: ignore[arg-type]

    def _neg_fraction(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return sum(1 for value in values if value <= 0.0) / float(len(values))

    return {
        "row_neg_fraction": _neg_fraction(center_row),
        "row_min": min(center_row),
        "col_neg_fraction": _neg_fraction(center_col),
        "col_min": min(center_col),
    }


def compute_rollout_metrics(
    predicted_runs: Sequence[Sequence[Sequence[Sequence[float]]]],
    reference_runs: Sequence[SyntheticSDFRun],
) -> dict[str, float]:
    if not predicted_runs or not reference_runs:
        raise ValueError("predicted/reference runs are empty")

    l1_total = 0.0
    l2_total = 0.0
    num_points = 0
    num_steps = 0
    num_runs = 0

    vn_abs_total = 0.0
    vn_sq_total = 0.0
    vn_count = 0

    profile_err_total = {
        "row_neg_fraction": 0.0,
        "row_min": 0.0,
        "col_neg_fraction": 0.0,
        "col_min": 0.0,
    }
    profile_count = 0
    obs_pred_frames: list[list[list[float]]] = []
    obs_ref_frames: list[list[list[float]]] = []

    for pred_phi_t, run in zip(predicted_runs, reference_runs):
        step_count = min(len(pred_phi_t), len(run.phi_t))
        if step_count == 0:
            continue
        num_runs += 1
        num_steps += step_count

        for step_index in range(step_count):
            pred_frame = pred_phi_t[step_index]
            ref_frame = run.phi_t[step_index]
            pred_vals = _flatten_values(pred_frame)
            ref_vals = _flatten_values(ref_frame)
            for pred_cell, ref_cell in zip(pred_vals, ref_vals):
                diff = float(pred_cell) - float(ref_cell)
                l1_total += abs(diff)
                l2_total += diff * diff
                num_points += 1

            if step_index + 1 < step_count:
                pred_next = pred_phi_t[step_index + 1]
                ref_next = run.phi_t[step_index + 1]
                pred_vn = (_frame_mean(pred_frame) - _frame_mean(pred_next)) / float(run.dt)
                ref_vn = (_frame_mean(ref_frame) - _frame_mean(ref_next)) / float(run.dt)
                vn_diff = pred_vn - ref_vn
                vn_abs_total += abs(vn_diff)
                vn_sq_total += vn_diff * vn_diff
                vn_count += 1

        pred_profile = _profile_features(pred_phi_t[step_count - 1])
        ref_profile = _profile_features(run.phi_t[step_count - 1])
        for key in profile_err_total:
            profile_err_total[key] += abs(pred_profile[key] - ref_profile[key])
        profile_count += 1
        obs_pred_frames.append([[float(cell) for cell in row] for row in pred_phi_t[step_count - 1]])
        obs_ref_frames.append([[float(cell) for cell in row] for row in run.phi_t[step_count - 1]])

    if num_points == 0:
        raise ValueError("no comparable points for rollout metrics")

    profile_norm = float(profile_count) if profile_count else 1.0
    vn_norm = float(vn_count) if vn_count else 1.0
    observation_metrics = compute_observation_feature_metrics(
        observation_model=BaselineSdfObservationModel(),
        predicted_shapes=obs_pred_frames,
        reference_shapes=obs_ref_frames,
    )
    shape_aliases: dict[str, float] = {}
    for src_key, alias in (
        ("obs_cd_top_ratio_mae", "shape_cd_top_mae"),
        ("obs_cd_mid_ratio_mae", "shape_cd_mid_mae"),
        ("obs_cd_bottom_ratio_mae", "shape_cd_bottom_mae"),
        ("obs_sidewall_angle_deg_mae", "shape_sidewall_angle_mae"),
        ("obs_centerline_curvature_proxy_mae", "shape_curvature_proxy_mae"),
        ("obs_footing_proxy_mae", "shape_footing_proxy_mae"),
    ):
        value = observation_metrics.get(src_key)
        if isinstance(value, (int, float)):
            shape_aliases[alias] = float(value)

    return {
        "sdf_l1_mean": l1_total / float(num_points),
        "sdf_l2_rmse": math.sqrt(l2_total / float(num_points)),
        "vn_mae": vn_abs_total / vn_norm,
        "vn_rmse": math.sqrt(vn_sq_total / vn_norm),
        "profile_row_neg_fraction_mae": profile_err_total["row_neg_fraction"] / profile_norm,
        "profile_row_min_mae": profile_err_total["row_min"] / profile_norm,
        "profile_col_neg_fraction_mae": profile_err_total["col_neg_fraction"] / profile_norm,
        "profile_col_min_mae": profile_err_total["col_min"] / profile_norm,
        "num_points_compared": float(num_points),
        "num_steps_compared": float(num_steps),
        "num_runs_compared": float(num_runs),
        **observation_metrics,
        **shape_aliases,
    }


def format_metrics_json(metrics: Mapping[str, float]) -> str:
    return json.dumps(dict(metrics), sort_keys=True)


def compute_temporal_diagnostics(
    *,
    predicted_phi_t: Sequence[object],
    reference_phi_t: Sequence[object],
) -> dict[str, object]:
    step_count = min(len(predicted_phi_t), len(reference_phi_t))
    if step_count < 2:
        return {
            "num_steps": int(step_count),
            "delta_phi_sign_agreement": 0.0,
            "delta_phi_mae_per_step": [],
            "early_window_error": 0.0,
            "late_window_error": 0.0,
            "r2_all_frames": 0.0,
            "r2_final_frame": 0.0,
        }

    pred_means = [_frame_mean(predicted_phi_t[idx]) for idx in range(step_count)]
    ref_means = [_frame_mean(reference_phi_t[idx]) for idx in range(step_count)]
    pred_delta = [float(pred_means[idx + 1] - pred_means[idx]) for idx in range(step_count - 1)]
    ref_delta = [float(ref_means[idx + 1] - ref_means[idx]) for idx in range(step_count - 1)]
    delta_abs = [abs(float(p) - float(t)) for p, t in zip(pred_delta, ref_delta)]

    sign_hits = 0
    sign_total = 0
    for pred_value, ref_value in zip(pred_delta, ref_delta):
        if abs(float(ref_value)) < 1e-12:
            continue
        sign_total += 1
        if (float(pred_value) >= 0.0) == (float(ref_value) >= 0.0):
            sign_hits += 1
    sign_agreement = float(sign_hits) / float(sign_total) if sign_total else 0.0

    early_cut = min(3, len(delta_abs))
    early_errors = delta_abs[:early_cut]
    late_errors = delta_abs[early_cut:]
    early_window_error = float(fmean(early_errors)) if early_errors else 0.0
    late_window_error = float(fmean(late_errors)) if late_errors else 0.0

    all_pred_vals: list[float] = []
    all_ref_vals: list[float] = []
    for idx in range(step_count):
        p_vals = _flatten_values(predicted_phi_t[idx])
        t_vals = _flatten_values(reference_phi_t[idx])
        n = min(len(p_vals), len(t_vals))
        for point_idx in range(n):
            all_pred_vals.append(float(p_vals[point_idx]))
            all_ref_vals.append(float(t_vals[point_idx]))

    def _r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        if not y_true:
            return 0.0
        mean_true = fmean(y_true)
        ss_res = sum((float(t) - float(p)) ** 2 for t, p in zip(y_true, y_pred))
        ss_tot = sum((float(t) - float(mean_true)) ** 2 for t in y_true)
        if ss_tot <= 1e-12:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))

    final_pred = _flatten_values(predicted_phi_t[step_count - 1])
    final_ref = _flatten_values(reference_phi_t[step_count - 1])
    final_n = min(len(final_pred), len(final_ref))

    return {
        "num_steps": int(step_count),
        "delta_phi_sign_agreement": float(sign_agreement),
        "delta_phi_mae_per_step": [float(value) for value in delta_abs],
        "early_window_error": float(early_window_error),
        "late_window_error": float(late_window_error),
        "r2_all_frames": float(_r2_score(all_ref_vals, all_pred_vals)),
        "r2_final_frame": float(_r2_score(final_ref[:final_n], final_pred[:final_n])),
    }
