from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from wafer_surrogate.observation.plugins.baseline import (
    _as_2d_frame,
    _as_sequence,
    _is_nested_sequence,
    _project_frame_features,
)
from wafer_surrogate.observation.registry import ObservationModel, ObservationModelError, ShapeState, register_observation_model


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(float(v) for v in values) / float(len(values))
    var = sum((float(v) - mean) ** 2 for v in values) / float(len(values))
    return float(mean), float(math.sqrt(max(0.0, var)))


@dataclass(frozen=True)
class MultiViewSdfObservationModel:
    num_views: int = 5
    top_depth_ratio: float = 0.2
    mid_depth_ratio: float = 0.5
    bottom_depth_ratio: float = 0.8
    sdf_threshold: float = 0.0
    narrow_band_width: float = 0.5
    _base_feature_names: tuple[str, ...] = (
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

    def feature_names(self) -> list[str]:
        out: list[str] = []
        for name in self._base_feature_names:
            out.append(f"{name}_mean")
            out.append(f"{name}_std")
        return out

    def _view_indices(self, depth: int) -> list[int]:
        if depth < 1:
            return []
        n = max(1, min(int(self.num_views), depth))
        if n == 1:
            return [depth // 2]
        return sorted({int(round(i * (depth - 1) / float(n - 1))) for i in range(n)})

    def project(self, shape_state: ShapeState) -> list[float]:
        outer = _as_sequence("shape_state", shape_state)
        if not outer:
            raise ObservationModelError("shape_state must be non-empty")

        first = _as_sequence("shape_state[0]", outer[0])
        if not first:
            raise ObservationModelError("shape_state nested sequence must be non-empty")

        views: list[list[list[float]]] = []
        if _is_nested_sequence(first[0]):
            for z_idx in self._view_indices(len(outer)):
                views.append(_as_2d_frame(outer[z_idx]))  # type: ignore[arg-type]
        else:
            views = [_as_2d_frame(shape_state)]

        projected = [
            _project_frame_features(
                frame=view,
                sdf_threshold=float(self.sdf_threshold),
                top_depth_ratio=float(self.top_depth_ratio),
                mid_depth_ratio=float(self.mid_depth_ratio),
                bottom_depth_ratio=float(self.bottom_depth_ratio),
                narrow_band_width=float(self.narrow_band_width),
            )
            for view in views
        ]
        if not projected:
            raise ObservationModelError("multiview projection produced no features")

        dim = len(projected[0])
        out: list[float] = []
        for d_idx in range(dim):
            col = [float(row[d_idx]) for row in projected]
            mean, std = _mean_std(col)
            out.extend([mean, std])
        return out


@register_observation_model("multiview")
def _build_multiview_observation_model(
    num_views: int = 5,
    top_depth_ratio: float = 0.2,
    mid_depth_ratio: float = 0.5,
    bottom_depth_ratio: float = 0.8,
    sdf_threshold: float = 0.0,
    narrow_band_width: float = 0.5,
) -> ObservationModel:
    return MultiViewSdfObservationModel(
        num_views=max(1, int(num_views)),
        top_depth_ratio=float(top_depth_ratio),
        mid_depth_ratio=float(mid_depth_ratio),
        bottom_depth_ratio=float(bottom_depth_ratio),
        sdf_threshold=float(sdf_threshold),
        narrow_band_width=float(narrow_band_width),
    )
