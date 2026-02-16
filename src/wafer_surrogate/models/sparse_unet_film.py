from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wafer_surrogate.geometry import finite_diff_grad

try:  # Optional dependencies.
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - environment dependent
    torch = None
    nn = None

try:  # Optional dependency.
    import MinkowskiEngine as ME
except Exception:  # pragma: no cover - environment dependent
    ME = None


class OptionalSparseDependencyUnavailable(RuntimeError):
    """Raised when sparse tensor dependencies are unavailable."""


def sparse_dependencies_available() -> bool:
    return torch is not None and nn is not None and ME is not None


def _require_sparse_dependencies() -> tuple[Any, Any, Any]:
    if torch is None or nn is None or ME is None:
        missing: list[str] = []
        if torch is None or nn is None:
            missing.append("torch")
        if ME is None:
            missing.append("MinkowskiEngine")
        raise OptionalSparseDependencyUnavailable(
            "sparse tensor training requires optional dependencies: " + ", ".join(missing)
        )
    return torch, nn, ME


def normalize_condition_scaler(
    payload: Mapping[str, Any] | None,
    *,
    cond_dim: int,
) -> dict[str, Any]:
    dim = max(0, int(cond_dim))
    default = {
        "schema_version": 1,
        "enabled": False,
        "mean": [0.0 for _ in range(dim)],
        "std": [1.0 for _ in range(dim)],
    }
    if payload is None:
        return default
    raw_mean = payload.get("mean")
    raw_std = payload.get("std")
    if not isinstance(raw_mean, list) or not isinstance(raw_std, list):
        return default
    mean = [float(v) for v in raw_mean[:dim]]
    std = [float(v) for v in raw_std[:dim]]
    while len(mean) < dim:
        mean.append(0.0)
    while len(std) < dim:
        std.append(1.0)
    std = [1.0 if abs(v) < 1e-12 else float(v) for v in std]
    return {
        "schema_version": 1,
        "enabled": bool(payload.get("enabled", True)),
        "mean": mean,
        "std": std,
    }


def _apply_condition_scaler(values: list[float], condition_scaler: Mapping[str, Any] | None) -> list[float]:
    if not condition_scaler or not bool(condition_scaler.get("enabled", False)):
        return list(values)
    mean_raw = condition_scaler.get("mean")
    std_raw = condition_scaler.get("std")
    if not isinstance(mean_raw, list) or not isinstance(std_raw, list):
        return list(values)
    out: list[float] = []
    for idx, value in enumerate(values):
        mu = float(mean_raw[idx]) if idx < len(mean_raw) else 0.0
        sigma = float(std_raw[idx]) if idx < len(std_raw) else 1.0
        if abs(sigma) < 1e-12:
            sigma = 1.0
        out.append((float(value) - mu) / sigma)
    return out


@dataclass(frozen=True)
class FeatureContract:
    recipe_keys: list[str]
    feature_names: list[str]
    cond_dim: int
    feat_dim: int
    band_width: float = 0.5
    min_grad_norm: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_keys": list(self.recipe_keys),
            "feature_names": list(self.feature_names),
            "cond_dim": int(self.cond_dim),
            "feat_dim": int(self.feat_dim),
            "band_width": float(self.band_width),
            "min_grad_norm": float(self.min_grad_norm),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureContract":
        return cls(
            recipe_keys=[str(v) for v in payload.get("recipe_keys", [])],
            feature_names=[str(v) for v in payload.get("feature_names", [])],
            cond_dim=int(payload.get("cond_dim", 0)),
            feat_dim=int(payload.get("feat_dim", 0)),
            band_width=float(payload.get("band_width", 0.5)),
            min_grad_norm=float(payload.get("min_grad_norm", 1e-6)),
        )


def condition_vector(
    conditions: Mapping[str, float],
    contract: FeatureContract,
    *,
    condition_scaler: Mapping[str, Any] | None = None,
) -> list[float]:
    values: list[float] = []
    if contract.recipe_keys:
        values.extend(float(conditions.get(key, 0.0)) for key in contract.recipe_keys)
        if len(values) < contract.cond_dim:
            latent_keys = sorted(
                key for key in conditions.keys()
                if str(key).startswith("z_") and str(key) not in set(contract.recipe_keys)
            )
            values.extend(float(conditions[key]) for key in latent_keys)
    else:
        ordered = sorted((str(k), float(v)) for k, v in conditions.items())
        values.extend(float(v) for _, v in ordered)
    if len(values) < contract.cond_dim:
        values.extend([0.0] * (contract.cond_dim - len(values)))
    values = values[: contract.cond_dim]
    return _apply_condition_scaler(values, condition_scaler)


def _to_numpy(frame: Any) -> Any:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for sparse tensor feature extraction") from exc

    return np.asarray(frame, dtype=float)


def _coords_with_batch(coords_xyz: Any, batch_index: int) -> Any:
    np = __import__("numpy")
    if coords_xyz.size == 0:
        return np.zeros((0, 4), dtype=np.int32)
    batch_col = np.full((coords_xyz.shape[0], 1), int(batch_index), dtype=np.int32)
    return np.concatenate([batch_col, coords_xyz.astype(np.int32)], axis=1)


def build_narrow_band_sparse_inputs(
    phi: Any,
    *,
    conditions: dict[str, float],
    step_index: int,
    contract: FeatureContract,
    condition_scaler: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, list[float]]:
    np = __import__("numpy")
    phi_arr = _to_numpy(phi)
    grad = finite_diff_grad(phi_arr)
    grad_norm = np.linalg.norm(grad, axis=0)

    curvature = np.zeros_like(phi_arr, dtype=float)
    if phi_arr.ndim == 2:
        h, w = phi_arr.shape
        for y in range(h):
            for x in range(w):
                c = float(phi_arr[y, x])
                l = float(phi_arr[y, x - 1]) if x > 0 else c
                r = float(phi_arr[y, x + 1]) if x + 1 < w else c
                u = float(phi_arr[y - 1, x]) if y > 0 else c
                d = float(phi_arr[y + 1, x]) if y + 1 < h else c
                curvature[y, x] = (l + r + u + d) - (4.0 * c)
    elif phi_arr.ndim == 3:
        dz, h, w = phi_arr.shape
        for z in range(dz):
            for y in range(h):
                for x in range(w):
                    c = float(phi_arr[z, y, x])
                    l = float(phi_arr[z, y, x - 1]) if x > 0 else c
                    r = float(phi_arr[z, y, x + 1]) if x + 1 < w else c
                    u = float(phi_arr[z, y - 1, x]) if y > 0 else c
                    d = float(phi_arr[z, y + 1, x]) if y + 1 < h else c
                    pz = float(phi_arr[z - 1, y, x]) if z > 0 else c
                    nz = float(phi_arr[z + 1, y, x]) if z + 1 < dz else c
                    curvature[z, y, x] = (l + r + u + d + pz + nz) - (6.0 * c)

    mask = (np.abs(phi_arr) <= float(contract.band_width)) & (grad_norm >= float(contract.min_grad_norm))
    coords = np.argwhere(mask)
    if coords.size == 0:
        shape = phi_arr.shape
        center = tuple(int(dim // 2) for dim in shape)
        coords = np.array([center], dtype=np.int32)

    feat_rows: list[list[float]] = []
    for coord in coords:
        phi_value = float(phi_arr[tuple(coord)])
        if coords.shape[1] == 2:
            x = float(coord[1])
            y = float(coord[0])
            z = 0.0
        elif coords.shape[1] == 3:
            x = float(coord[2])
            y = float(coord[1])
            z = float(coord[0])
        else:
            raise ValueError(f"unsupported phi rank for sparse inputs: {coords.shape[1]}")
        feature_map = {
            "nb_feat_0": phi_value,
            "nb_feat_1": float(grad_norm[tuple(coord)]),
            "nb_feat_2": float(curvature[tuple(coord)]),
            "nb_feat_3": abs(phi_value),
            "phi": phi_value,
            "grad_norm": float(grad_norm[tuple(coord)]),
            "grad_abs": float(grad_norm[tuple(coord)]),
            "curvature_proxy": float(curvature[tuple(coord)]),
            "band_distance": abs(phi_value),
            "coord_x": x,
            "coord_y": y,
            "coord_z": z,
            "step_index": float(step_index),
        }
        if contract.feature_names:
            row = [float(feature_map.get(name, 0.0)) for name in contract.feature_names]
        else:
            row = [phi_value, x, y, z, float(step_index)]
            if len(row) < contract.feat_dim:
                row.extend([0.0] * (contract.feat_dim - len(row)))
            row = row[: contract.feat_dim]
        feat_rows.append(row)

    cond_vec = condition_vector(conditions, contract, condition_scaler=condition_scaler)

    # Ensure xyz coords even for 2D grids.
    if coords.shape[1] == 2:
        coords_xyz = np.concatenate([coords[:, [1]], coords[:, [0]], np.zeros((coords.shape[0], 1), dtype=np.int32)], axis=1)
    elif coords.shape[1] == 3:
        coords_xyz = np.concatenate([coords[:, [2]], coords[:, [1]], coords[:, [0]]], axis=1)
    else:
        raise ValueError(f"unsupported phi rank for sparse inputs: {coords.shape[1]}")

    return coords_xyz.astype(np.int32), np.asarray(feat_rows, dtype=np.float32), cond_vec


def splat_vn_to_grid(
    *,
    phi_shape: tuple[int, ...],
    coords_xyz: Any,
    vn_values: Any,
    default_value: float,
) -> Any:
    np = __import__("numpy")
    out = np.full(phi_shape, float(default_value), dtype=float)
    if coords_xyz.size == 0:
        return out

    vn = np.asarray(vn_values, dtype=float).reshape(-1)
    n = min(len(vn), int(coords_xyz.shape[0]))
    for idx in range(n):
        x = int(coords_xyz[idx, 0])
        y = int(coords_xyz[idx, 1])
        z = int(coords_xyz[idx, 2])
        if len(phi_shape) == 2:
            if 0 <= y < phi_shape[0] and 0 <= x < phi_shape[1]:
                out[y, x] = float(vn[idx])
        else:
            if 0 <= z < phi_shape[0] and 0 <= y < phi_shape[1] and 0 <= x < phi_shape[2]:
                out[z, y, x] = float(vn[idx])
    return out


class SparseUNetFiLM(nn.Module if nn is not None else object):
    def __init__(
        self,
        in_channels: int,
        cond_dim: int,
        hidden_channels: int = 32,
        num_blocks: int = 2,
        dropout: float = 0.0,
        residual: bool = True,
        out_channels: int = 1,
    ) -> None:
        torch_mod, nn_mod, me_mod = _require_sparse_dependencies()
        super().__init__()
        self.in_channels = int(in_channels)
        self.cond_dim = int(cond_dim)
        self.hidden_channels = int(hidden_channels)
        self.num_blocks = max(1, int(num_blocks))
        self.dropout = max(0.0, float(dropout))
        self.residual = bool(residual)

        self.convs = nn_mod.ModuleList()
        self.bns = nn_mod.ModuleList()
        self.films = nn_mod.ModuleList()
        in_ch = self.in_channels
        for _ in range(self.num_blocks):
            self.convs.append(
                me_mod.MinkowskiConvolution(
                    in_channels=in_ch,
                    out_channels=self.hidden_channels,
                    kernel_size=3,
                    stride=1,
                    dimension=3,
                )
            )
            self.bns.append(me_mod.MinkowskiBatchNorm(self.hidden_channels))
            self.films.append(
                nn_mod.Sequential(
                    nn_mod.Linear(max(1, self.cond_dim), self.hidden_channels * 2),
                    nn_mod.ReLU(),
                    nn_mod.Linear(self.hidden_channels * 2, self.hidden_channels * 2),
                )
            )
            in_ch = self.hidden_channels
        self.head = me_mod.MinkowskiConvolution(in_channels=self.hidden_channels, out_channels=out_channels, kernel_size=1, stride=1, dimension=3)
        self.relu = me_mod.MinkowskiReLU()
        self.dropout_layer = nn_mod.Dropout(self.dropout) if self.dropout > 0.0 else None

    def _apply_film(self, sparse_tensor: Any, params: Any) -> Any:
        torch_mod, _, me_mod = _require_sparse_dependencies()
        gamma, beta = torch_mod.chunk(params, chunks=2, dim=1)
        batch_idx = sparse_tensor.C[:, 0].long()
        features = sparse_tensor.F * (1.0 + gamma[batch_idx]) + beta[batch_idx]
        return me_mod.SparseTensor(
            features,
            coordinate_map_key=sparse_tensor.coordinate_map_key,
            coordinate_manager=sparse_tensor.coordinate_manager,
        )

    def _apply_dropout(self, sparse_tensor: Any) -> Any:
        _, _, me_mod = _require_sparse_dependencies()
        if self.dropout_layer is None:
            return sparse_tensor
        features = self.dropout_layer(sparse_tensor.F)
        return me_mod.SparseTensor(
            features,
            coordinate_map_key=sparse_tensor.coordinate_map_key,
            coordinate_manager=sparse_tensor.coordinate_manager,
        )

    def _apply_residual(self, x: Any, skip: Any) -> Any:
        _, _, me_mod = _require_sparse_dependencies()
        if not self.residual:
            return x
        if int(skip.F.shape[1]) != int(x.F.shape[1]):
            return x
        return me_mod.SparseTensor(
            x.F + skip.F,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )

    def forward(self, sparse_tensor: Any, cond_vec: Any, *, return_features: bool = False) -> Any:
        x = sparse_tensor
        for block_idx in range(self.num_blocks):
            skip = x
            x = self.convs[block_idx](x)
            x = self.bns[block_idx](x)
            x = self._apply_film(x, self.films[block_idx](cond_vec))
            x = self.relu(x)
            x = self._apply_dropout(x)
            x = self._apply_residual(x, skip)

        mid_features = x.F
        out = self.head(x)
        if return_features:
            return out, mid_features
        return out


class SparseTensorVnModel:
    def __init__(
        self,
        *,
        network: Any,
        contract: FeatureContract,
        device: str = "cpu",
        architecture: Mapping[str, Any] | None = None,
        condition_scaler: Mapping[str, Any] | None = None,
    ) -> None:
        _require_sparse_dependencies()
        self.network = network
        self.contract = contract
        self.device = str(device)
        self.architecture = {str(k): v for k, v in (architecture or {}).items()}
        self.condition_scaler = normalize_condition_scaler(
            condition_scaler,
            cond_dim=int(contract.cond_dim),
        )

    def predict(self, features: dict[str, float]) -> float:
        values = [float(v) for v in features.values()]
        return float(sum(values) / float(len(values))) if values else 0.0

    def predict_vn(self, phi: Any, conditions: dict[str, float], step_index: int = 0) -> Any:
        torch_mod, _, me_mod = _require_sparse_dependencies()
        coords_xyz, feat_rows, cond_vec = build_narrow_band_sparse_inputs(
            phi,
            conditions={str(k): float(v) for k, v in conditions.items()},
            step_index=int(step_index),
            contract=self.contract,
            condition_scaler=self.condition_scaler,
        )
        if feat_rows.size == 0:
            np = __import__("numpy")
            return np.zeros_like(_to_numpy(phi), dtype=float)

        coords_b = _coords_with_batch(coords_xyz, 0)
        coords_t = torch_mod.as_tensor(coords_b, dtype=torch_mod.int32, device=self.device)
        feat_t = torch_mod.as_tensor(feat_rows, dtype=torch_mod.float32, device=self.device)
        cond_t = torch_mod.as_tensor([cond_vec], dtype=torch_mod.float32, device=self.device)

        st = me_mod.SparseTensor(features=feat_t, coordinates=coords_t)
        self.network.eval()
        with torch_mod.no_grad():
            out = self.network(st, cond_t)
        vn_sparse = out.F.detach().cpu().numpy().reshape(-1)
        default_value = float(vn_sparse.mean()) if vn_sparse.size > 0 else 0.0
        np = __import__("numpy")
        phi_shape = tuple(int(v) for v in _to_numpy(phi).shape)
        return splat_vn_to_grid(
            phi_shape=phi_shape,
            coords_xyz=coords_xyz,
            vn_values=vn_sparse,
            default_value=default_value,
        )

    def encode_conditions(self, conditions: Mapping[str, float]) -> list[float]:
        return condition_vector(
            {str(key): float(value) for key, value in conditions.items()},
            self.contract,
            condition_scaler=self.condition_scaler,
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        torch_mod, _, _ = _require_sparse_dependencies()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.network.state_dict(),
            "contract": self.contract.to_dict(),
            "architecture": dict(self.architecture),
            "condition_scaler": dict(self.condition_scaler),
        }
        torch_mod.save(payload, str(out))
        return out

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str = "cpu") -> "SparseTensorVnModel":
        torch_mod, _, _ = _require_sparse_dependencies()
        payload = torch_mod.load(str(path), map_location=device)
        if not isinstance(payload, dict):
            raise ValueError("invalid sparse checkpoint payload")
        contract_raw = payload.get("contract")
        if not isinstance(contract_raw, dict):
            raise ValueError("sparse checkpoint missing contract")
        contract = FeatureContract.from_dict(contract_raw)
        arch_raw = payload.get("architecture")
        arch = {str(k): v for k, v in arch_raw.items()} if isinstance(arch_raw, dict) else {}
        scaler_raw = payload.get("condition_scaler")
        scaler = scaler_raw if isinstance(scaler_raw, Mapping) else None
        network = SparseUNetFiLM(
            in_channels=max(1, int(contract.feat_dim)),
            cond_dim=max(1, int(contract.cond_dim)),
            hidden_channels=max(1, int(arch.get("hidden_channels", 32))),
            num_blocks=max(1, int(arch.get("num_blocks", 2))),
            dropout=max(0.0, float(arch.get("dropout", 0.0))),
            residual=bool(arch.get("residual", True)),
            out_channels=1,
        )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("sparse checkpoint missing state_dict")
        network.load_state_dict(state_dict)
        network.to(device)
        return cls(
            network=network,
            contract=contract,
            device=device,
            architecture=arch,
            condition_scaler=scaler,
        )
