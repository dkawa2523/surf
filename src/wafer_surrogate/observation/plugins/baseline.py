from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from wafer_surrogate.observation.registry import ObservationModel, ObservationModelError, ShapeState, register_observation_model


def _as_sequence(name: str, value: object) -> list[object]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ObservationModelError(f"{name} must be a sequence")
    return list(value)


def _as_float(name: str, value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ObservationModelError(f"{name} must be numeric") from exc
    if not math.isfinite(out):
        raise ObservationModelError(f"{name} must be finite")
    return out


def _is_nested_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _as_2d_frame(shape_state: ShapeState) -> list[list[float]]:
    outer = _as_sequence("shape_state", shape_state)
    if not outer:
        raise ObservationModelError("shape_state must be non-empty")

    first = outer[0]
    if not _is_nested_sequence(first):
        raise ObservationModelError("shape_state must be a 2D or 3D nested sequence")

    first_seq = _as_sequence("shape_state[0]", first)
    if not first_seq:
        raise ObservationModelError("shape_state nested sequence must be non-empty")

    if _is_nested_sequence(first_seq[0]):
        mid_depth = len(outer) // 2
        return _as_2d_frame(outer[mid_depth])  # type: ignore[arg-type]

    width: int | None = None
    frame: list[list[float]] = []
    for row_idx, row_obj in enumerate(outer):
        row_seq = _as_sequence(f"shape_state[{row_idx}]", row_obj)
        if width is None:
            width = len(row_seq)
            if width == 0:
                raise ObservationModelError("shape_state rows must be non-empty")
        elif len(row_seq) != width:
            raise ObservationModelError("shape_state rows must share the same width")
        frame.append(
            [_as_float(f"shape_state[{row_idx}][{col_idx}]", cell) for col_idx, cell in enumerate(row_seq)]
        )
    return frame


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ObservationModelError("feature source sequence must be non-empty")
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return mean, math.sqrt(max(0.0, variance))


def _row_negative_ratio(row: Sequence[float], threshold: float = 0.0) -> float:
    if not row:
        return 0.0
    return sum(1 for value in row if float(value) <= float(threshold)) / float(len(row))


def _finite_diff_grad_abs(frame: Sequence[Sequence[float]]) -> list[float]:
    ny = len(frame)
    nx = len(frame[0]) if ny else 0
    if nx == 0 or ny == 0:
        return []

    values: list[float] = []
    for y in range(ny):
        for x in range(nx):
            left = float(frame[y][x - 1]) if x > 0 else float(frame[y][x])
            right = float(frame[y][x + 1]) if (x + 1) < nx else float(frame[y][x])
            up = float(frame[y - 1][x]) if y > 0 else float(frame[y][x])
            down = float(frame[y + 1][x]) if (y + 1) < ny else float(frame[y][x])
            dx = 0.5 * (right - left)
            dy = 0.5 * (down - up)
            values.append(math.sqrt((dx * dx) + (dy * dy)))
    return values


def _centerline_curvature_proxy(frame: Sequence[Sequence[float]]) -> float:
    ny = len(frame)
    if ny < 3:
        return 0.0
    cx = len(frame[0]) // 2
    col = [float(frame[y][cx]) for y in range(ny)]
    if len(col) < 3:
        return 0.0
    second_diff = [abs(col[i + 1] - (2.0 * col[i]) + col[i - 1]) for i in range(1, len(col) - 1)]
    return sum(second_diff) / float(len(second_diff))


def _project_frame_features(
    *,
    frame: list[list[float]],
    sdf_threshold: float,
    top_depth_ratio: float,
    mid_depth_ratio: float,
    bottom_depth_ratio: float,
    narrow_band_width: float,
) -> list[float]:
    center_row = frame[len(frame) // 2]
    center_col = [row[len(row) // 2] for row in frame]
    flat = [value for row in frame for value in row]
    ny = len(frame)

    mean_2d, std_2d = _mean_std(flat)
    row_mean, row_std = _mean_std(center_row)
    col_mean, col_std = _mean_std(center_col)
    grad_abs = _finite_diff_grad_abs(frame)
    grad_abs_mean, _ = _mean_std(grad_abs) if grad_abs else (0.0, 0.0)
    grad_abs_max = max(grad_abs) if grad_abs else 0.0
    narrow_band_ratio = sum(1 for value in flat if abs(float(value)) <= float(narrow_band_width)) / float(len(flat))

    top_idx = min(max(int(round((ny - 1) * float(top_depth_ratio))), 0), ny - 1)
    mid_idx = min(max(int(round((ny - 1) * float(mid_depth_ratio))), 0), ny - 1)
    bottom_idx = min(max(int(round((ny - 1) * float(bottom_depth_ratio))), 0), ny - 1)

    cd_top = _row_negative_ratio(frame[top_idx], threshold=float(sdf_threshold))
    cd_mid = _row_negative_ratio(frame[mid_idx], threshold=float(sdf_threshold))
    cd_bottom = _row_negative_ratio(frame[bottom_idx], threshold=float(sdf_threshold))
    depth_span = max(1, bottom_idx - top_idx)
    sidewall_slope = abs(cd_bottom - cd_top) / float(depth_span)
    sidewall_angle_deg = math.degrees(math.atan(sidewall_slope))
    footing_proxy = max(0.0, cd_bottom - cd_mid)
    center_curvature = _centerline_curvature_proxy(frame)

    return [
        float(mean_2d),
        float(std_2d),
        float(min(flat)),
        float(max(flat)),
        float(row_mean),
        float(row_std),
        float(col_mean),
        float(col_std),
        float(cd_top),
        float(cd_mid),
        float(cd_bottom),
        float(sidewall_angle_deg),
        float(center_curvature),
        float(footing_proxy),
        float(grad_abs_mean),
        float(grad_abs_max),
        float(narrow_band_ratio),
    ]


@dataclass(frozen=True)
class BaselineSdfObservationModel:
    _feature_names: tuple[str, ...] = (
        "phi_mean_2d",
        "phi_std_2d",
        "phi_min_2d",
        "phi_max_2d",
        "center_row_mean_1d",
        "center_row_std_1d",
        "center_col_mean_1d",
        "center_col_std_1d",
        "cd_top_ratio",
        "cd_mid_ratio",
        "cd_bottom_ratio",
        "sidewall_angle_deg",
        "centerline_curvature_proxy",
        "footing_proxy",
        "grad_abs_mean_2d",
        "grad_abs_max_2d",
        "narrow_band_ratio_2d",
    )
    top_depth_ratio: float = 0.2
    mid_depth_ratio: float = 0.5
    bottom_depth_ratio: float = 0.8
    sdf_threshold: float = 0.0
    narrow_band_width: float = 0.5

    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def project(self, shape_state: ShapeState) -> list[float]:
        outer = _as_sequence("shape_state", shape_state)
        if not outer:
            raise ObservationModelError("shape_state must be non-empty")
        first = _as_sequence("shape_state[0]", outer[0])
        if not first:
            raise ObservationModelError("shape_state nested sequence must be non-empty")

        use_volume = _is_nested_sequence(first[0])
        frames: list[list[list[float]]] = []
        if use_volume:
            depth = len(outer)
            anchor = sorted({0, max(0, depth // 4), max(0, depth // 2), max(0, (3 * depth) // 4), max(0, depth - 1)})
            for z_idx in anchor:
                frames.append(_as_2d_frame(outer[z_idx]))  # type: ignore[arg-type]
        else:
            frames = [_as_2d_frame(shape_state)]

        projected = [
            _project_frame_features(
                frame=frame,
                sdf_threshold=float(self.sdf_threshold),
                top_depth_ratio=float(self.top_depth_ratio),
                mid_depth_ratio=float(self.mid_depth_ratio),
                bottom_depth_ratio=float(self.bottom_depth_ratio),
                narrow_band_width=float(self.narrow_band_width),
            )
            for frame in frames
        ]
        if len(projected) == 1:
            return projected[0]
        dim = len(projected[0])
        return [
            float(sum(projected[s_idx][d_idx] for s_idx in range(len(projected))) / float(len(projected)))
            for d_idx in range(dim)
        ]


@register_observation_model("baseline")
def _build_baseline_observation_model(
    top_depth_ratio: float = 0.2,
    mid_depth_ratio: float = 0.5,
    bottom_depth_ratio: float = 0.8,
    sdf_threshold: float = 0.0,
    narrow_band_width: float = 0.5,
) -> ObservationModel:
    return BaselineSdfObservationModel(
        top_depth_ratio=float(top_depth_ratio),
        mid_depth_ratio=float(mid_depth_ratio),
        bottom_depth_ratio=float(bottom_depth_ratio),
        sdf_threshold=float(sdf_threshold),
        narrow_band_width=float(narrow_band_width),
    )

