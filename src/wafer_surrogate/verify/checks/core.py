from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path


def _normalize_argv(argv: Sequence[str] | None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--"]:
        return args[1:]
    return args


def _check_import(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, f"ok: import {module_name}"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: import {module_name} ({exc})"


def _check_registry_behaviors() -> tuple[bool, str]:
    try:
        from wafer_surrogate.features import list_feature_extractors, make_feature_extractor
        from wafer_surrogate.models import list_models, make_model
        from wafer_surrogate.preprocess import (
            build_preprocess_pipeline,
            list_preprocess_steps,
        )

        extractors = list_feature_extractors()
        if not extractors:
            return False, "fail: feature extractor registry is empty"
        sample_extractor = make_feature_extractor(extractors[0])
        sample_features = sample_extractor.extract({"x": 1.0, "y": 2.0})
        if not sample_features:
            return False, "fail: feature extractor returned no features"

        preprocess_steps = list_preprocess_steps()
        required_steps = {"offset", "scale"}
        if not required_steps.issubset(set(preprocess_steps)):
            return False, "fail: preprocess registry missing required baseline steps"

        first_order = build_preprocess_pipeline(
            [
                ("offset", {"field": "x", "value": 2.0}),
                ("scale", {"field": "x", "factor": 3.0}),
            ]
        ).transform({"x": 1.0})
        second_order = build_preprocess_pipeline(
            [
                ("scale", {"field": "x", "factor": 3.0}),
                ("offset", {"field": "x", "value": 2.0}),
            ]
        ).transform({"x": 1.0})
        if first_order["x"] == second_order["x"]:
            return False, "fail: preprocess pipeline order is not respected"

        models = list_models()
        required_models = {
            "baseline_mean",
            "baseline_vn_constant",
            "baseline_vn_linear",
            "baseline_vn_linear_trainable",
            "operator_time_conditioned",
            "sparse_vn_student",
            "sparse_vn_teacher",
            "surface_graph_vn",
        }
        if not required_models.issubset(set(models)):
            return False, "fail: baseline model registry is missing required entries"
        baseline_model = make_model("baseline_mean", default_value=1.0)
        baseline_model.fit([{"x": 1.0}, {"x": 2.0}], [2.0, 4.0])
        if baseline_model.predict({"x": 0.0}) != 3.0:
            return False, "fail: baseline model prediction is incorrect"

        linear_model = make_model(
            "baseline_vn_linear",
            default_value=1.0,
            condition_weights={"pressure": 0.01},
        )
        linear_pred = linear_model.predict({"pressure": 20.0})
        if abs(linear_pred - 1.2) > 1e-12:
            return False, "fail: linear baseline condition term is incorrect"

        trainable_linear_model = make_model(
            "baseline_vn_linear_trainable",
            learning_rate=0.05,
            epochs=120,
        )
        trainable_linear_model.fit(
            [{"x": 0.0}, {"x": 1.0}, {"x": 2.0}],
            [0.5, 1.5, 2.5],
        )
        if abs(trainable_linear_model.predict({"x": 1.5}) - 2.0) > 5e-2:
            return False, "fail: trainable linear model fit is incorrect"

        return True, "ok: registries and baseline components"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: registry behavior ({exc})"


def _check_synthetic_generation() -> tuple[bool, str]:
    try:
        from wafer_surrogate.data.synthetic import (
            generate_synthetic_sdf_dataset,
            write_synthetic_example,
        )

        dataset = generate_synthetic_sdf_dataset(
            num_runs=1,
            num_steps=3,
            grid_size=8,
            dt=0.1,
        )
        if len(dataset.runs) != 1:
            return False, "fail: synthetic dataset run count is incorrect"
        run = dataset.runs[0]
        if len(run.phi_t) != 3:
            return False, "fail: synthetic phi(t) length is incorrect"
        if len(run.phi_t[0]) != 8 or len(run.phi_t[0][0]) != 8:
            return False, "fail: synthetic grid shape is incorrect"

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = write_synthetic_example(
                output_dir=tmp_dir,
                num_runs=1,
                num_steps=3,
                grid_size=8,
                dt=0.1,
            )
            with output_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            if not payload.get("runs"):
                return False, "fail: synthetic example output is empty"
        return True, "ok: synthetic SDF data generation"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: synthetic generation ({exc})"


def _check_vn_pseudo_label_generation() -> tuple[bool, str]:
    try:
        from wafer_surrogate.data.io import (
            InMemoryAdapter,
            read_narrow_band_dataset,
            synthetic_to_narrow_band_dataset,
            write_narrow_band_dataset,
        )
        from wafer_surrogate.data.synthetic import SyntheticSDFDataset, SyntheticSDFRun
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: pseudo-label import ({exc})"

    try:
        plane_t = [
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ]
        plane_next = [
            [-1.2, -0.2, 0.8],
            [-1.2, -0.2, 0.8],
            [-1.2, -0.2, 0.8],
        ]
        dataset = SyntheticSDFDataset(
            runs=[
                SyntheticSDFRun(
                    run_id="pseudo_labels",
                    dt=0.1,
                    recipe={"pressure": 20.0},
                    phi_t=[plane_t, plane_next],
                )
            ]
        )
        narrow_band = synthetic_to_narrow_band_dataset(
            dataset,
            band_width=0.1,
            min_grad_norm=0.5,
        )
        step0 = narrow_band.runs[0].steps[0]
        if len(step0.coords) != 3:
            return False, "fail: pseudo-label generation narrow-band row count mismatch"
        for target in step0.vn_target:
            if abs(float(target[0]) - 2.0) > 1e-6:
                return False, "fail: pseudo-label value mismatch"

        masked_dataset = SyntheticSDFDataset(
            runs=[
                SyntheticSDFRun(
                    run_id="masked",
                    dt=0.1,
                    recipe={"pressure": 20.0},
                    phi_t=[
                        [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                        ],
                        [
                            [-0.1, -0.1, -0.1],
                            [-0.1, -0.1, -0.1],
                            [-0.1, -0.1, -0.1],
                        ],
                    ],
                )
            ]
        )
        masked = synthetic_to_narrow_band_dataset(
            masked_dataset,
            band_width=0.2,
            min_grad_norm=1e-6,
        )
        if masked.runs[0].steps[0].coords:
            return False, "fail: stability mask did not remove low-gradient samples"

        adapter = InMemoryAdapter()
        write_narrow_band_dataset(adapter, narrow_band)
        loaded = read_narrow_band_dataset(adapter)
        loaded_step0 = loaded.runs[0].steps[0]
        if len(loaded_step0.coords) != 3 or len(loaded_step0.vn_target) != 3:
            return False, "fail: pseudo-label save/load row count mismatch"

        return True, "ok: pseudo-label generation + stability mask"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: pseudo-label behavior ({exc})"


def _check_geometry_utilities() -> tuple[bool, str]:
    try:
        from wafer_surrogate.geometry import (
            extract_narrow_band,
            finite_diff_grad,
            levelset_step,
        )
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: geometry import ({exc})"

    try:
        import numpy as np
    except Exception:
        return True, "skip(optional): geometry unit-like checks require numpy"

    try:
        # phi(x, y) = x - 2 gives a constant x-gradient magnitude of about 1.
        phi = np.tile(np.arange(5, dtype=float), (5, 1)) - 2.0
        coords, values = extract_narrow_band(phi, band_width=0.25)
        if coords.shape != (5, 2) or values.shape != (5,):
            return False, "fail: extract_narrow_band returned unexpected sparse shape"

        grad = finite_diff_grad(phi)
        if grad.shape != (2, 5, 5):
            return False, "fail: finite_diff_grad returned unexpected shape"
        if not np.isclose(float(grad[0, 2, 2]), 0.0, atol=1e-6):
            return False, "fail: finite_diff_grad y-component mismatch"
        if not np.isclose(float(grad[1, 2, 2]), 1.0, atol=1e-6):
            return False, "fail: finite_diff_grad x-component mismatch"

        vn = np.ones_like(phi) * 0.2
        phi_next = levelset_step(phi, vn, dt=0.5)
        expected_center = float(phi[2, 2]) - 0.1
        if not np.isclose(float(phi_next[2, 2]), expected_center, atol=1e-6):
            return False, "fail: levelset_step update mismatch"
        return True, "ok: geometry utilities"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: geometry behavior ({exc})"


def _check_simulate_rollout() -> tuple[bool, str]:
    try:
        from wafer_surrogate.inference.simulate import simulate
        from wafer_surrogate.models import make_model
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: simulate import ({exc})"

    try:
        import numpy as np
    except Exception:
        return True, "skip(optional): simulate rollout checks require numpy"

    try:
        # Keep phi simple/planar so rollout behavior is easy to validate.
        phi0 = np.tile(np.arange(6, dtype=float), (6, 1)) - 2.5
        model = make_model("baseline_vn_constant", default_value=0.2)
        phi_t = simulate(
            model=model,
            phi0=phi0,
            conditions={"pressure": 20.0},
            num_steps=4,
            dt=0.1,
        )
        if len(phi_t) != 4:
            return False, "fail: simulate returned unexpected time-series length"
        if phi_t[0].shape != phi_t[-1].shape:
            return False, "fail: simulate changed grid shape"

        expected_center = float(phi0[3, 3]) - 0.2 * 0.1 * 3
        if not np.isclose(float(phi_t[-1][3, 3]), expected_center, atol=1e-6):
            return False, "fail: simulate rollout update mismatch"
        return True, "ok: simulate rollout"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: simulate behavior ({exc})"


def _check_surface_graph_model() -> tuple[bool, str]:
    try:
        from wafer_surrogate.data.synthetic import generate_synthetic_sdf_dataset
        from wafer_surrogate.inference.simulate import simulate
        from wafer_surrogate.models import make_model
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: surface graph import ({exc})"

    try:
        import numpy as np
    except Exception:
        return True, "skip(optional): surface graph checks require numpy"

    try:
        dataset = generate_synthetic_sdf_dataset(
            num_runs=1,
            num_steps=4,
            grid_size=20,
            dt=0.1,
        )
        run = dataset.runs[0]
        phi0 = np.asarray(run.phi_t[0], dtype=float)
        conditions = {str(key): float(value) for key, value in run.recipe.items()}

        graph_model = make_model(
            "surface_graph_vn",
            default_value=0.03,
            condition_weights={"pressure": 0.001},
            local_strength=0.04,
            nonlocal_strength=0.45,
            local_radius=2.0,
            nonlocal_radius=8.5,
            max_nonlocal_degree=10,
        )
        vn_graph = np.asarray(
            graph_model.predict_vn(phi=phi0, conditions=conditions, step_index=0),
            dtype=float,
        )
        if vn_graph.shape != phi0.shape:
            return False, "fail: surface graph vn shape mismatch"
        if not np.all(np.isfinite(vn_graph)):
            return False, "fail: surface graph vn contains non-finite values"

        local_only_model = make_model(
            "surface_graph_vn",
            default_value=0.03,
            condition_weights={"pressure": 0.001},
            local_strength=0.04,
            nonlocal_strength=0.0,
            local_radius=2.0,
            nonlocal_radius=8.5,
            max_nonlocal_degree=10,
        )
        vn_local = np.asarray(
            local_only_model.predict_vn(phi=phi0, conditions=conditions, step_index=0),
            dtype=float,
        )
        band = np.abs(phi0) <= 0.9
        if int(np.count_nonzero(band)) < 8:
            return False, "fail: synthetic surface band is too small for graph checks"
        coupling_delta = float(np.mean(np.abs(vn_graph[band] - vn_local[band])))
        if coupling_delta <= 1e-6:
            return False, "fail: non-local coupling has no measurable effect on Vn"
        if float(np.mean(vn_graph[band])) <= 0.0:
            return False, "fail: predicted Vn is not physically plausible near the surface"

        rollout = simulate(
            model=graph_model,
            phi0=phi0,
            conditions=conditions,
            num_steps=4,
            dt=float(run.dt),
        )
        if len(rollout) != 4:
            return False, "fail: surface graph rollout length mismatch"
        if float(np.mean(rollout[-1])) >= float(np.mean(rollout[0])):
            return False, "fail: surface graph rollout did not progress in etch direction"
        return True, "ok: surface graph model + SDF projection"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: surface graph behavior ({exc})"


def _check_operator_baseline() -> tuple[bool, str]:
    try:
        from wafer_surrogate.data.synthetic import generate_synthetic_sdf_dataset
        from wafer_surrogate.models import make_model
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: operator baseline import ({exc})"

    try:
        import numpy as np
    except Exception:
        return True, "skip(optional): operator baseline checks require numpy"

    try:
        dataset = generate_synthetic_sdf_dataset(
            num_runs=3,
            num_steps=6,
            grid_size=12,
            dt=0.1,
        )
        run = dataset.runs[0]
        model = make_model("operator_time_conditioned", default_value=0.0, l2=1e-8)
        if not hasattr(model, "fit_operator") or not hasattr(model, "predict_phi"):
            return False, "fail: operator model missing fit_operator/predict_phi"

        model.fit_operator(dataset.runs)
        phi0 = np.asarray(run.phi_t[0], dtype=float)

        # Query a non-grid time (between training snapshots).
        queried = np.asarray(
            model.predict_phi(
                phi0=phi0,
                conditions=run.recipe,
                t=0.25,
            ),
            dtype=float,
        )
        if queried.shape != phi0.shape:
            return False, "fail: operator baseline phi(t) shape mismatch"
        if not np.all(np.isfinite(queried)):
            return False, "fail: operator baseline phi(t) has non-finite values"

        lower = np.asarray(run.phi_t[2], dtype=float)  # t=0.2
        upper = np.asarray(run.phi_t[3], dtype=float)  # t=0.3
        expected_mid = 0.5 * (lower + upper)
        mae = float(np.mean(np.abs(queried - expected_mid)))
        if mae > 5e-2:
            return False, "fail: operator baseline interpolation error too large"

        offgrid = np.asarray(
            model.predict_phi(
                phi0=phi0,
                conditions=run.recipe,
                t=0.37,
            ),
            dtype=float,
        )
        if offgrid.shape != phi0.shape:
            return False, "fail: operator baseline arbitrary-time query failed"
        return True, "ok: operator baseline train + arbitrary-time query"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: operator baseline behavior ({exc})"


def _check_metrics_and_ood_hooks() -> tuple[bool, str]:
    try:
        from wafer_surrogate.data.synthetic import generate_synthetic_sdf_dataset
        from wafer_surrogate.inference.ood import assess_ood
        from wafer_surrogate.metrics import compute_rollout_metrics
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: metrics/ood import ({exc})"

    try:
        dataset = generate_synthetic_sdf_dataset(
            num_runs=2,
            num_steps=3,
            grid_size=8,
            dt=0.1,
        )
        predicted = [run.phi_t for run in dataset.runs]
        metrics = compute_rollout_metrics(predicted_runs=predicted, reference_runs=dataset.runs)
        if float(metrics.get("sdf_l1_mean", 1.0)) > 1e-12:
            return False, "fail: metrics sdf_l1_mean mismatch on identical rollout"
        if float(metrics.get("vn_mae", 1.0)) > 1e-12:
            return False, "fail: metrics vn_mae mismatch on identical rollout"

        ood = assess_ood(
            query_conditions=dataset.runs[0].recipe,
            reference_conditions=[run.recipe for run in dataset.runs],
            threshold=3.0,
        )
        if not bool(ood.get("in_domain", False)):
            return False, "fail: ood hook in-domain check mismatch"
        return True, "ok: metrics + ood hooks"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: metrics/ood behavior ({exc})"


def _check_sem_io_and_map_calibration() -> tuple[bool, str]:
    try:
        from wafer_surrogate.data.sem import load_sem_features
        from wafer_surrogate.inference.calibrate import calibrate_latent_map
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: sem/calibration import ({exc})"

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = f"{tmp_dir}/sem_features.json"
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump({"feature_names": ["cd_top", "angle"], "y": [12.5, 88.0]}, fp)
            sem_json = load_sem_features(json_path)
            if sem_json.feature_names != ["cd_top", "angle"]:
                return False, "fail: sem json feature names mismatch"
            if sem_json.y != [12.5, 88.0]:
                return False, "fail: sem json values mismatch"

            csv_path = f"{tmp_dir}/sem_features.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(["cd_bottom", "curvature"])
                writer.writerow([6.2, 0.45])
            sem_csv = load_sem_features(csv_path)
            if sem_csv.feature_names != ["cd_bottom", "curvature"]:
                return False, "fail: sem csv feature names mismatch"
            if any(abs(a - b) > 1e-12 for a, b in zip(sem_csv.y, [6.2, 0.45])):
                return False, "fail: sem csv values mismatch"

        true_z = [0.6, -0.4]

        def synthetic_predict_features(z: Sequence[float]) -> list[float]:
            z0 = float(z[0])
            z1 = float(z[1])
            return [
                1.2 + 2.0 * z0 - 0.5 * z1,
                -0.8 + 0.25 * z0 + 1.5 * z1,
            ]

        target_y = synthetic_predict_features(true_z)
        result = calibrate_latent_map(
            predict_features=synthetic_predict_features,
            target_y=target_y,
            z_init=[0.0, 0.0],
            prior_mean=[0.0, 0.0],
            prior_std=[100.0, 100.0],
            learning_rate=0.4,
            max_iters=120,
            tol=1e-10,
        )
        if max(abs(a - b) for a, b in zip(result.y_pred, target_y)) > 1e-4:
            return False, "fail: MAP calibration did not match target SEM features"
        if max(abs(a - b) for a, b in zip(result.z_map, true_z)) > 6e-2:
            return False, "fail: MAP calibration latent z mismatch"
        return True, "ok: sem feature I/O + MAP latent calibration"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: sem/calibration behavior ({exc})"


def _check_observation_projection_and_calibration() -> tuple[bool, str]:
    try:
        from wafer_surrogate.data.synthetic import generate_synthetic_sdf_dataset
        from wafer_surrogate.inference.calibrate import calibrate_latent_map_with_observation
        from wafer_surrogate.observation import (
            BaselineSdfObservationModel,
            compute_observation_feature_metrics,
        )
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: observation import ({exc})"

    try:
        dataset = generate_synthetic_sdf_dataset(
            num_runs=1,
            num_steps=3,
            grid_size=8,
            dt=0.1,
        )
        frame = dataset.runs[0].phi_t[-1]
        obs_model = BaselineSdfObservationModel()

        first = obs_model.project(frame)
        second = obs_model.project(frame)
        if first != second:
            return False, "fail: observation baseline projection is not deterministic"
        if len(first) != len(obs_model.feature_names()):
            return False, "fail: observation feature names and output dimensions mismatch"

        obs_eval = compute_observation_feature_metrics(
            observation_model=obs_model,
            predicted_shapes=[frame],
            reference_shapes=[frame],
        )
        if float(obs_eval.get("obs_feature_mae", 1.0)) > 1e-12:
            return False, "fail: observation eval mae mismatch on identical shapes"
        if float(obs_eval.get("obs_feature_rmse", 1.0)) > 1e-12:
            return False, "fail: observation eval rmse mismatch on identical shapes"

        def simulate_shape(z: Sequence[float]) -> list[list[float]]:
            shift = float(z[0])
            return [[float(cell) - shift for cell in row] for row in frame]

        target_z = [0.25]
        target_y = obs_model.project(simulate_shape(target_z))
        result = calibrate_latent_map_with_observation(
            simulate_shape=simulate_shape,
            observation_model=obs_model,
            target_y=target_y,
            z_init=[0.0],
            prior_mean=[0.0],
            prior_std=[100.0],
            learning_rate=0.2,
            max_iters=100,
            tol=1e-10,
        )
        if abs(result.z_map[0] - target_z[0]) > 2e-2:
            return False, "fail: observation-based MAP calibration latent mismatch"
        if max(abs(a - b) for a, b in zip(result.y_pred, target_y)) > 1e-4:
            return False, "fail: observation-based MAP calibration feature mismatch"

        return True, "ok: observation projection + calibration/eval hooks"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: observation behavior ({exc})"


def _check_shape_prior_hooks() -> tuple[bool, str]:
    try:
        from wafer_surrogate.prior import list_shape_priors, make_shape_prior
        from wafer_surrogate.workflows.prior_utils import build_prior
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: shape prior import ({exc})"

    try:
        if "gaussian_latent" not in list_shape_priors():
            return False, "fail: gaussian_latent prior is not registered"

        prior = make_shape_prior("gaussian_latent", latent_dim=2, mean=[0.0, 0.0], std=[1.0, 2.0])
        samples_a = prior.sample_latent(num_samples=2, seed=123)
        samples_b = prior.sample_latent(num_samples=2, seed=123)
        if samples_a != samples_b:
            return False, "fail: gaussian prior sampling is not reproducible with fixed seed"
        if len(samples_a) != 2 or any(len(vec) != 2 for vec in samples_a):
            return False, "fail: gaussian prior sample shape mismatch"
        score = prior.score_latent(samples_a[0])
        if not isinstance(score, float):
            return False, "fail: gaussian prior score must return float"

        prior_name, prior_kwargs, loaded_prior = build_prior(
            {
                "prior": {
                    "name": "gaussian_latent",
                    "latent_dim": 2,
                    "mean": [0.0, 0.0],
                    "std": [1.0, 2.0],
                }
            }
        )
        if prior_name != "gaussian_latent":
            return False, "fail: config prior loader did not keep requested prior name"
        if int(prior_kwargs.get("latent_dim", 0)) != 2:
            return False, "fail: config prior loader latent_dim mismatch"
        loaded_sample = loaded_prior.sample_latent(num_samples=1, seed=0)[0]
        if len(loaded_sample) != 2:
            return False, "fail: config prior loader sample dimension mismatch"

        return True, "ok: shape prior hooks + config loader wiring"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: shape prior behavior ({exc})"


def _check_sbi_posterior_estimation() -> tuple[bool, str]:
    try:
        from wafer_surrogate.inference.calibrate import (
            OptionalDependencyUnavailable,
            sample_latent_posterior_sbi,
            train_latent_posterior_sbi,
        )
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: sbi posterior import ({exc})"

    try:
        latent_samples: list[list[float]] = []
        observations: list[list[float]] = []
        for i in range(5):
            for j in range(5):
                z0 = -1.0 + (0.5 * float(i))
                z1 = -1.0 + (0.5 * float(j))
                latent_samples.append([z0, z1])
                observations.append(
                    [
                        0.2 + 1.4 * z0 - 0.35 * z1,
                        -0.5 + 0.25 * z0 + 1.1 * z1,
                    ]
                )

        estimator = train_latent_posterior_sbi(
            latent_samples=latent_samples,
            observations=observations,
            training_batch_size=16,
            max_num_epochs=8,
        )
        target_z = [0.35, -0.2]
        target_obs = [
            0.2 + 1.4 * target_z[0] - 0.35 * target_z[1],
            -0.5 + 0.25 * target_z[0] + 1.1 * target_z[1],
        ]
        sampled = sample_latent_posterior_sbi(
            estimator=estimator,
            observation=target_obs,
            num_samples=48,
            seed=0,
        )
        if len(sampled) != 48:
            return False, "fail: SBI posterior sample count mismatch"
        mean_z0 = sum(float(z[0]) for z in sampled) / float(len(sampled))
        mean_z1 = sum(float(z[1]) for z in sampled) / float(len(sampled))
        if abs(mean_z0 - target_z[0]) > 0.5 or abs(mean_z1 - target_z[1]) > 0.5:
            return False, "fail: SBI posterior samples do not track latent target"
        return True, "ok: SBI posterior train + sample"
    except OptionalDependencyUnavailable as exc:
        return True, f"skip(optional): {exc}"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: sbi posterior behavior ({exc})"


def _check_teacher_student_distillation() -> tuple[bool, str]:
    try:
        from wafer_surrogate.pipeline import run_pipeline
        from wafer_surrogate.runtime import detect_runtime_capabilities
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: distillation import ({exc})"

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            suffix = str(random.randint(10000, 99999))
            cfg_path = Path(tmp_dir) / "verify_distill.toml"
            cfg_path.write_text(
                f"""
[run]
output_dir = "runs"

[data]
source = "synthetic"
num_runs = 2
num_steps = 4
grid_size = 10
dt = 0.1

[pipeline]
run_id = "verify_distill_{suffix}"
stages = ["cleaning", "featurization", "preprocessing", "train"]

[featurization]
target_mode = "vn_narrow_band"
nb_backend = "memory"
emit_priv = true
priv_source = "proxy"

[preprocessing]
steps = ["identity"]

[train]
mode = "sparse_distill"
fallback_model = "baseline_vn_linear_trainable"
sparse_batch_points = 300
seed = 0
""".strip(),
                encoding="utf-8",
            )
            manifest = run_pipeline(
                config_path=cfg_path,
                selected_stages=["cleaning", "featurization", "preprocessing", "train"],
            )
            if manifest.stage_status.get("train") != "ok":
                return False, "fail: train stage status is not ok for sparse_distill"
            run_dir = Path("runs") / manifest.run_id
            train_metrics_path = run_dir / "train" / "outputs" / "train_metrics.json"
            if not train_metrics_path.exists():
                return False, "fail: train_metrics.json missing for sparse_distill"
            with train_metrics_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            if not isinstance(payload, dict):
                return False, "fail: train_metrics.json is not a mapping"
            best_metrics = payload.get("best_metrics")
            if not isinstance(best_metrics, dict):
                return False, "fail: best_metrics missing in train_metrics.json"
            student_mae = best_metrics.get("student_mae", best_metrics.get("mae"))
            if student_mae is None:
                return False, "fail: student_mae/mae missing in sparse_distill metrics"
            if detect_runtime_capabilities().sparse_backend:
                if not (run_dir / "train" / "outputs" / "distill_metrics.json").exists():
                    return False, "fail: distill_metrics.json missing under sparse backend"
            elif not any("sparse_distill fallback" in str(message) for message in manifest.warnings):
                return False, "fail: sparse_distill fallback warning missing in manifest"
        return True, "ok: teacher-student distillation (pipeline train stage)"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: distillation behavior ({exc})"


def _check_pipeline_stage_workflow(*, include_inference: bool = True) -> tuple[bool, str]:
    try:
        from wafer_surrogate.pipeline import run_pipeline
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: pipeline import ({exc})"

    try:
        def _check_featurization_bundle(run_id: str) -> tuple[bool, str]:
            bundle_path = Path("runs") / run_id / "featurization" / "outputs" / "reconstruction_bundle.json"
            if not bundle_path.exists():
                return False, "fail: featurization reconstruction_bundle.json was not written"
            with bundle_path.open("r", encoding="utf-8") as fp:
                bundle = json.load(fp)
            if not isinstance(bundle, dict):
                return False, "fail: featurization reconstruction bundle is not a mapping"
            if not str(bundle.get("payload_path", "")).endswith("/features.csv"):
                return False, "fail: featurization reconstruction bundle payload_path is invalid"
            if not str(bundle.get("target_path", "")).endswith("/targets.csv"):
                return False, "fail: featurization reconstruction bundle target_path is invalid"
            refs = bundle.get("row_refs")
            if not isinstance(refs, list) or not refs:
                return False, "fail: featurization reconstruction bundle row_refs are missing"
            first_ref = refs[0]
            required_ref_keys = {"sample_index", "run_id", "run_index", "step_index", "dt"}
            if not isinstance(first_ref, dict) or not required_ref_keys.issubset(set(first_ref.keys())):
                return False, "fail: featurization reconstruction bundle row_refs schema is invalid"
            hooks = bundle.get("post_inference_hooks")
            if not isinstance(hooks, list) or not hooks:
                return False, "fail: featurization reconstruction bundle post_inference_hooks are missing"
            inverse_mapping = bundle.get("inverse_mapping")
            if not isinstance(inverse_mapping, dict):
                return False, "fail: featurization reconstruction bundle inverse_mapping is missing"
            return True, "ok"

        def _check_preprocess_bundle(run_id: str) -> tuple[bool, str]:
            bundle_path = Path("runs") / run_id / "preprocessing" / "outputs" / "preprocess_bundle.json"
            if not bundle_path.exists():
                return False, "fail: preprocessing preprocess_bundle.json was not written"
            with bundle_path.open("r", encoding="utf-8") as fp:
                bundle = json.load(fp)
            if not isinstance(bundle, dict):
                return False, "fail: preprocessing bundle is not a mapping"
            if "schema_version" not in bundle:
                return False, "fail: preprocessing bundle schema_version is missing"
            normalization = bundle.get("normalization")
            if not isinstance(normalization, dict):
                return False, "fail: preprocessing bundle normalization is missing"
            inverse_metadata = bundle.get("inverse_metadata")
            if not isinstance(inverse_metadata, dict):
                return False, "fail: preprocessing bundle inverse_metadata is missing"
            inverse_mapping = bundle.get("inverse_mapping")
            if not isinstance(inverse_mapping, dict):
                return False, "fail: preprocessing bundle inverse_mapping is missing"
            if "feature_contract_hash" not in bundle:
                return False, "fail: preprocessing bundle feature_contract_hash is missing"
            if "inverse_ready" not in bundle:
                return False, "fail: preprocessing bundle inverse_ready is missing"
            report_path = Path("runs") / run_id / "preprocessing" / "outputs" / "preprocess_report.json"
            if not report_path.exists():
                return False, "fail: preprocessing preprocess_report.json was not written"
            with report_path.open("r", encoding="utf-8") as fp:
                report = json.load(fp)
            if not isinstance(report, dict):
                return False, "fail: preprocessing report payload is not a mapping"
            if "schema_version" not in report:
                return False, "fail: preprocessing report schema_version is missing"
            if not isinstance(report.get("reconstruction_error"), dict):
                return False, "fail: preprocessing report reconstruction_error is missing"
            feature_stats = report.get("feature_stats")
            if not isinstance(feature_stats, dict):
                return False, "fail: preprocessing report feature_stats is missing"
            return True, "ok"

        def _check_manifest_schema(run_id: str) -> tuple[bool, str]:
            manifest_path = Path("runs") / run_id / "manifest.json"
            if not manifest_path.exists():
                return False, "fail: manifest.json was not written"
            with manifest_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            if not isinstance(payload, dict):
                return False, "fail: manifest payload is not a mapping"
            required_keys = {
                "schema_version",
                "stage_status",
                "stage_dependencies",
                "stage_inputs",
                "stage_artifacts",
                "stage_metrics",
                "runtime_env",
                "seed_info",
                "split_info",
                "warnings",
            }
            if not required_keys.issubset(set(payload.keys())):
                return False, "fail: manifest is missing required schema keys"
            split_info = payload.get("split_info")
            if not isinstance(split_info, dict):
                return False, "fail: manifest split_info is not a mapping"
            required_split_keys = {
                "num_train_runs",
                "num_valid_runs",
                "leak_checked",
                "reason",
                "loader_mode",
            }
            stage_status = payload.get("stage_status")
            train_enabled = isinstance(stage_status, dict) and str(stage_status.get("train", "")) == "ok"
            if train_enabled and not required_split_keys.issubset(set(split_info.keys())):
                return False, "fail: manifest split_info missing required keys"
            return True, "ok"

        def _check_leaderboard_viz(run_id: str) -> tuple[bool, str]:
            viz_manifest = Path("runs") / run_id / "leaderboard" / "viz" / "leaderboard_plot_manifest.json"
            if not viz_manifest.exists():
                return False, "fail: leaderboard_plot_manifest.json was not written"
            with viz_manifest.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            if not isinstance(payload, dict):
                return False, "fail: leaderboard_plot_manifest payload is not a mapping"
            ranking_priority = payload.get("ranking_priority")
            if not isinstance(ranking_priority, list) or not ranking_priority:
                return False, "fail: leaderboard_plot_manifest missing ranking_priority"
            expected = ["metric_student_mae", "metric_rollout_short_window_error", "score"]
            if [str(v) for v in ranking_priority] != expected:
                return False, "fail: leaderboard_plot_manifest ranking_priority is invalid"
            return True, "ok"

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = f"{tmp_dir}/pipeline_verify.toml"
            with open(cfg_path, "w", encoding="utf-8") as fp:
                fp.write(
                    """
[run]
output_dir = "runs"

[data]
source = "synthetic"
num_runs = 1
num_steps = 4
grid_size = 10
dt = 0.1

[featurization]
extractor = "identity"

[preprocessing]
steps = ["identity"]
normalize = true

[train]
model_name = "baseline_vn_linear_trainable"

[inference]
mode = "single"
ood_threshold = 3.0
""".strip()
                )
            primary_stages = ["cleaning", "featurization", "preprocessing", "train"]
            if include_inference:
                primary_stages.append("inference")

            manifest = run_pipeline(
                config_path=cfg_path,
                selected_stages=primary_stages,
            )
            if not manifest.stage_order:
                return False, "fail: pipeline produced empty stage_order"
            for stage in primary_stages:
                if manifest.stage_status.get(stage) != "ok":
                    return False, f"fail: pipeline stage '{stage}' status is not ok"
            if not (Path("runs") / manifest.run_id / "manifest.json").exists():
                return False, "fail: pipeline manifest file was not written"
            manifest_ok, manifest_message = _check_manifest_schema(manifest.run_id)
            if not manifest_ok:
                return False, manifest_message
            viz_ok, viz_message = _check_leaderboard_viz(manifest.run_id)
            if not viz_ok:
                return False, viz_message
            bundle_ok, bundle_message = _check_featurization_bundle(manifest.run_id)
            if not bundle_ok:
                return False, bundle_message
            preprocess_ok, preprocess_message = _check_preprocess_bundle(manifest.run_id)
            if not preprocess_ok:
                return False, preprocess_message

            skip_stages = ["featurization", "preprocessing", "train"]
            if include_inference:
                skip_stages.append("inference")

            skip_manifest = run_pipeline(
                config_path=cfg_path,
                selected_stages=skip_stages,
            )
            if "cleaning" in skip_manifest.stage_order:
                return False, "fail: pipeline skip-cleaning run unexpectedly included cleaning stage"
            for stage in skip_stages:
                if skip_manifest.stage_status.get(stage) != "ok":
                    return False, f"fail: pipeline skip-cleaning stage '{stage}' status is not ok"
            viz_ok, viz_message = _check_leaderboard_viz(skip_manifest.run_id)
            if not viz_ok:
                return False, viz_message
            bundle_ok, bundle_message = _check_featurization_bundle(skip_manifest.run_id)
            if not bundle_ok:
                return False, bundle_message
            preprocess_ok, preprocess_message = _check_preprocess_bundle(skip_manifest.run_id)
            if not preprocess_ok:
                return False, preprocess_message

            preprocess_only_cfg_path = f"{tmp_dir}/pipeline_preprocess_only.toml"
            with open(preprocess_only_cfg_path, "w", encoding="utf-8") as fp:
                fp.write(
                    f"""
[run]
output_dir = "runs"

[data]
source = "synthetic"
num_runs = 1
num_steps = 4
grid_size = 10
dt = 0.1

[pipeline]
stages = ["preprocessing"]

[pipeline.external_inputs.preprocessing]
features_csv = "runs/{manifest.run_id}/featurization/outputs/features.csv"
targets_csv = "runs/{manifest.run_id}/featurization/outputs/targets.csv"
reconstruction_bundle_json = "runs/{manifest.run_id}/featurization/outputs/reconstruction_bundle.json"

[preprocessing]
steps = ["identity"]
normalize = true
""".strip()
                )
            preprocess_only_manifest = run_pipeline(
                config_path=preprocess_only_cfg_path,
                selected_stages=["preprocessing"],
            )
            if preprocess_only_manifest.stage_status.get("preprocessing") != "ok":
                return False, "fail: standalone preprocessing stage status is not ok"
            viz_ok, viz_message = _check_leaderboard_viz(preprocess_only_manifest.run_id)
            if not viz_ok:
                return False, viz_message
            preprocess_ok, preprocess_message = _check_preprocess_bundle(preprocess_only_manifest.run_id)
            if not preprocess_ok:
                return False, preprocess_message
            train_only_cfg_path = f"{tmp_dir}/pipeline_train_only.toml"
            with open(train_only_cfg_path, "w", encoding="utf-8") as fp:
                fp.write(
                    f"""
[run]
output_dir = "runs"

[data]
source = "synthetic"
num_runs = 1
num_steps = 4
grid_size = 10
dt = 0.1

[pipeline]
stages = ["train"]

[pipeline.external_inputs.train]
processed_features_csv = "runs/{manifest.run_id}/preprocessing/outputs/processed_features.csv"
processed_targets_csv = "runs/{manifest.run_id}/preprocessing/outputs/processed_targets.csv"
preprocess_bundle_json = "runs/{manifest.run_id}/preprocessing/outputs/preprocess_bundle.json"

[train]
model_name = "baseline_vn_linear_trainable"
cv_folds = 2
seed = 0
""".strip()
                )
            train_only_manifest = run_pipeline(
                config_path=train_only_cfg_path,
                selected_stages=["train"],
            )
            if train_only_manifest.stage_status.get("train") != "ok":
                return False, "fail: standalone train stage status is not ok"
            inference_only_cfg_path = f"{tmp_dir}/pipeline_infer_only.toml"
            with open(inference_only_cfg_path, "w", encoding="utf-8") as fp:
                fp.write(
                    f"""
[run]
output_dir = "runs"

[data]
source = "synthetic"
num_runs = 1
num_steps = 4
grid_size = 10
dt = 0.1

[pipeline]
stages = ["inference"]

[pipeline.external_inputs.inference]
model_state_json = "runs/{manifest.run_id}/train/outputs/model_state.json"
processed_features_csv = "runs/{manifest.run_id}/preprocessing/outputs/processed_features.csv"

[inference]
mode = "optimize"
engine = "optuna"
strategy = "random"
ood_threshold = 3.0
trials = 5
seed = 0
condition_ranges = {{ pressure = [0.0, 1.0], rf_power = [0.0, 1.0], gas_ratio = [0.0, 1.0] }}
""".strip()
                )
            inference_only_manifest = run_pipeline(
                config_path=inference_only_cfg_path,
                selected_stages=["inference"],
            )
            if inference_only_manifest.stage_status.get("inference") != "ok":
                return False, "fail: standalone inference stage status is not ok"
            optimize_summary_path = Path("runs") / inference_only_manifest.run_id / "inference" / "outputs" / "inference_optimize_summary.json"
            if not optimize_summary_path.exists():
                return False, "fail: standalone inference optimize summary is missing"
            with optimize_summary_path.open("r", encoding="utf-8") as fp:
                optimize_summary = json.load(fp)
            if not isinstance(optimize_summary, dict):
                return False, "fail: optimize summary is not a mapping"
            if str(optimize_summary.get("requested_engine", "")) != "optuna":
                return False, "fail: optimize summary requested_engine is invalid"
            if optimize_summary.get("resolved_engine") not in {"optuna", "builtin"}:
                return False, "fail: optimize summary resolved_engine is invalid"
            if optimize_summary.get("resolved_engine") == "builtin" and not optimize_summary.get("fallback_reason"):
                return False, "fail: optimize fallback reason is missing"
        if include_inference:
            return True, "ok: pipeline stage workflow run (cleaning->inference)"
        return True, "ok: pipeline stage workflow run (quick: cleaning->train)"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: pipeline workflow behavior ({exc})"


def _check_pipeline_advanced_features() -> tuple[bool, str]:
    try:
        from wafer_surrogate.pipeline import run_pipeline
        from wafer_surrogate.runtime import detect_runtime_capabilities
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: advanced pipeline import ({exc})"

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            suffix = str(random.randint(10000, 99999))
            sem_path = Path(tmp_dir) / "sem_features.json"
            with sem_path.open("w", encoding="utf-8") as fp:
                json.dump({"y": [0.0 for _ in range(17)]}, fp)

            cfg_path = Path(tmp_dir) / "pipeline_advanced.toml"
            cfg_path.write_text(
                f"""
[run]
output_dir = "runs"

[data]
source = "synthetic"
num_runs = 1
num_steps = 4
grid_size = 10
dt = 0.1

[pipeline]
run_id = "verify_adv_on_{suffix}"
stages = ["cleaning", "featurization", "preprocessing", "train", "inference"]

[featurization]
extractor = "identity"
target_mode = "vn_narrow_band"
band_width = 0.5
min_grad_norm = 1e-6
nb_backend = "memory"
emit_priv = true
priv_source = "auto"

[preprocessing]
steps = ["identity"]
normalize = true

[train]
mode = "sparse_distill"
teacher_model = "sparse_vn_teacher"
student_model = "sparse_vn_student"
fallback_model = "baseline_vn_linear_trainable"
sparse_batch_points = 400
seed = 0

[inference]
mode = "single"
reinit_enabled = true
reinit_every_n = 2
reinit_iters = 2
reinit_dt = 0.2

[inference.calibration]
enabled = true
method = "sbi"
sem_features = "{sem_path.as_posix()}"
latent_dim = 2
num_posterior_samples = 4
""".strip(),
                encoding="utf-8",
            )

            manifest = run_pipeline(config_path=cfg_path, selected_stages=["cleaning", "featurization", "preprocessing", "train", "inference"])
            for stage in ["cleaning", "featurization", "preprocessing", "train", "inference"]:
                if manifest.stage_status.get(stage) != "ok":
                    return False, f"fail: advanced pipeline stage '{stage}' status is not ok"

            run_dir = Path("runs") / manifest.run_id
            nb_manifest_path = run_dir / "featurization" / "outputs" / "narrow_band_manifest.json"
            if not nb_manifest_path.exists():
                return False, "fail: narrow_band_manifest.json is missing"
            with nb_manifest_path.open("r", encoding="utf-8") as fp:
                nb_manifest = json.load(fp)
            if str(nb_manifest.get("target_mode", "")) != "vn_narrow_band":
                return False, "fail: narrow_band_manifest target_mode is invalid"

            if detect_runtime_capabilities().sparse_backend:
                if not (run_dir / "train" / "outputs" / "distill_metrics.json").exists():
                    return False, "fail: distill_metrics.json is missing under sparse backend"
            else:
                if not any("sparse_distill fallback" in str(message) for message in manifest.warnings):
                    return False, "fail: sparse fallback warning is missing from manifest"

            if not any("priv_source auto fallback" in str(message) for message in manifest.warnings):
                return False, "fail: MC log auto/proxy fallback warning is missing"

            cal_map = run_dir / "inference" / "outputs" / "calibration_map.json"
            cal_sbi = run_dir / "inference" / "outputs" / "calibration_sbi_samples.json"
            if not cal_map.exists() and not cal_sbi.exists():
                return False, "fail: calibration artifact is missing"

            reinit_log = run_dir / "inference" / "outputs" / "inference_reinit_log.json"
            if not reinit_log.exists():
                return False, "fail: inference_reinit_log.json is missing"
            with reinit_log.open("r", encoding="utf-8") as fp:
                reinit_payload = json.load(fp)
            if int(reinit_payload.get("num_events", 0)) < 1:
                return False, "fail: reinit log did not record any event"

            # Reinit OFF baseline for behavioral switch check.
            cfg_off_path = Path(tmp_dir) / "pipeline_reinit_off.toml"
            cfg_off_path.write_text(
                f"""
[run]
output_dir = "runs"

[data]
source = "synthetic"
num_runs = 1
num_steps = 4
grid_size = 10
dt = 0.1

[pipeline]
run_id = "verify_adv_off_{suffix}"
stages = ["cleaning", "featurization", "preprocessing", "train", "inference"]

[featurization]
extractor = "identity"
target_mode = "frame_mean_delta"

[preprocessing]
steps = ["identity"]
normalize = true

[train]
model_name = "baseline_vn_linear_trainable"

[inference]
mode = "single"
reinit_enabled = false
""".strip(),
                encoding="utf-8",
            )
            manifest_off = run_pipeline(config_path=cfg_off_path, selected_stages=["cleaning", "featurization", "preprocessing", "train", "inference"])
            if manifest_off.stage_status.get("inference") != "ok":
                return False, "fail: reinit-off inference stage status is not ok"
            reinit_off_path = Path("runs") / manifest_off.run_id / "inference" / "outputs" / "inference_reinit_log.json"
            if reinit_off_path.exists():
                with reinit_off_path.open("r", encoding="utf-8") as fp:
                    reinit_off = json.load(fp)
                if int(reinit_off.get("num_events", 0)) != 0:
                    return False, "fail: reinit disabled run unexpectedly logged events"
        return True, "ok: advanced pipeline features (narrow-band/distill/calibration/reinit/proxy)"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: advanced pipeline behavior ({exc})"


def _check_contract_and_split_guards() -> tuple[bool, str]:
    try:
        from wafer_surrogate.core import (
            assert_feature_contract_compatible,
            validate_rows_against_feature_contract,
        )
        from wafer_surrogate.pipeline import run_pipeline
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: contract guard import ({exc})"

    try:
        # Feature contract mismatch must hard-fail.
        mismatch_failed = False
        try:
            assert_feature_contract_compatible(
                expected={
                    "feature_names": ["feat_a", "feat_b"],
                    "feat_dim": 2,
                    "cond_dim": 1,
                    "recipe_keys": ["recipe_0"],
                    "band_width": 0.5,
                    "min_grad_norm": 1e-6,
                },
                actual={
                    "feature_names": ["feat_b", "feat_a"],
                    "feat_dim": 2,
                    "cond_dim": 1,
                    "recipe_keys": ["recipe_0"],
                    "band_width": 0.5,
                    "min_grad_norm": 1e-6,
                },
                context="verify.contract",
            )
        except ValueError as exc:
            if "contract mismatch" not in str(exc):
                return False, "fail: feature contract mismatch error message is not standardized"
            mismatch_failed = True
        if not mismatch_failed:
            return False, "fail: feature contract mismatch did not raise"

        missing_feature_failed = False
        try:
            validate_rows_against_feature_contract(
                rows=[{"feat_a": 1.0}],
                contract={
                    "feature_names": ["feat_a", "feat_b"],
                    "feat_dim": 2,
                    "cond_dim": 1,
                    "recipe_keys": ["recipe_0"],
                    "band_width": 0.5,
                    "min_grad_norm": 1e-6,
                },
                source="verify.rows",
            )
        except ValueError as exc:
            if "contract mismatch" not in str(exc):
                return False, "fail: row/contract mismatch error message is not standardized"
            missing_feature_failed = True
        if not missing_feature_failed:
            return False, "fail: row/contract mismatch did not raise"

        order_mismatch_failed = False
        try:
            validate_rows_against_feature_contract(
                rows=[{"feat_b": 2.0, "feat_a": 1.0}],
                contract={
                    "feature_names": ["feat_a", "feat_b"],
                    "feat_dim": 2,
                    "cond_dim": 1,
                    "recipe_keys": ["recipe_0"],
                    "band_width": 0.5,
                    "min_grad_norm": 1e-6,
                },
                source="verify.rows.order",
            )
        except ValueError as exc:
            if "contract mismatch" not in str(exc):
                return False, "fail: row/contract order mismatch error message is not standardized"
            order_mismatch_failed = True
        if not order_mismatch_failed:
            return False, "fail: row/contract order mismatch did not raise"

        # strict_split=true must hard-fail when valid run split is impossible.
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "strict_split.toml"
            cfg_path.write_text(
                """
[run]
output_dir = "runs"

[data]
source = "synthetic"
num_runs = 1
num_steps = 4
grid_size = 10
dt = 0.1

[pipeline]
stages = ["cleaning", "featurization", "preprocessing", "train"]

[featurization]
extractor = "identity"
target_mode = "frame_mean_delta"

[preprocessing]
steps = ["identity"]
normalize = true

[train]
mode = "tabular"
model_name = "baseline_vn_linear_trainable"
strict_split = true
""".strip(),
                encoding="utf-8",
            )
            split_failed = False
            try:
                run_pipeline(
                    config_path=cfg_path,
                    selected_stages=["cleaning", "featurization", "preprocessing", "train"],
                )
            except Exception:
                split_failed = True
            if not split_failed:
                return False, "fail: strict_split=true with single run did not fail"
        return True, "ok: feature contract + strict split guards"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: contract/split guards ({exc})"


def _check_tracked_cache_artifacts() -> tuple[bool, str]:
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0 or probe.stdout.strip().lower() != "true":
            return True, "ok: tracked-cache check skipped (not a git worktree)"
        listed = subprocess.run(
            ["git", "ls-files"],
            check=False,
            capture_output=True,
            text=True,
        )
        if listed.returncode != 0:
            return False, "fail: unable to enumerate tracked files via git ls-files"
        offenders = []
        for raw in listed.stdout.splitlines():
            path = raw.strip()
            if not path:
                continue
            if "__pycache__/" in path or path.endswith(".pyc") or path.startswith(".venv_torchmd_tmp/"):
                offenders.append(path)
        if offenders:
            preview = ", ".join(offenders[:5])
            return False, f"fail: tracked cache artifacts detected ({preview})"
        return True, "ok: no tracked cache artifacts"
    except Exception as exc:  # pragma: no cover - defensive for environment issues
        return False, f"fail: tracked-cache check ({exc})"


def run_verify(quick: bool = False, full: bool = False) -> int:
    if full:
        os.environ.setdefault("WAFER_SURROGATE_REQUIRE_REAL_ME", "1")

    required_modules = [
        "wafer_surrogate",
        "wafer_surrogate.cli",
        "wafer_surrogate.data.synthetic",
        "wafer_surrogate.data.sem",
        "wafer_surrogate.features",
        "wafer_surrogate.geometry",
        "wafer_surrogate.inference",
        "wafer_surrogate.inference.calibrate",
        "wafer_surrogate.inference.ood",
        "wafer_surrogate.inference.simulate",
        "wafer_surrogate.metrics",
        "wafer_surrogate.observation",
        "wafer_surrogate.pipeline",
        "wafer_surrogate.prior",
        "wafer_surrogate.preprocess",
        "wafer_surrogate.config",
        "wafer_surrogate.models",
        "wafer_surrogate.models.api",
        "wafer_surrogate.verify",
    ]
    optional_modules = [
        "numpy",
        "matplotlib",
        "torch",
        "MinkowskiEngine",
        "sbi",
        "botorch",
    ]
    capabilities_obj = None
    try:
        from wafer_surrogate.runtime import detect_runtime_capabilities

        capabilities_obj = detect_runtime_capabilities()
        capabilities = capabilities_obj.missing_summary()
        print(f"runtime capabilities: {capabilities}")
    except Exception:
        pass

    failures = 0
    for module_name in required_modules:
        ok, message = _check_import(module_name)
        print(message)
        if not ok:
            failures += 1

    registry_ok, registry_message = _check_registry_behaviors()
    print(registry_message)
    if not registry_ok:
        failures += 1

    synthetic_ok, synthetic_message = _check_synthetic_generation()
    print(synthetic_message)
    if not synthetic_ok:
        failures += 1

    pseudo_ok, pseudo_message = _check_vn_pseudo_label_generation()
    print(pseudo_message)
    if not pseudo_ok:
        failures += 1

    geometry_ok, geometry_message = _check_geometry_utilities()
    print(geometry_message)
    if not geometry_ok:
        failures += 1

    simulate_ok, simulate_message = _check_simulate_rollout()
    print(simulate_message)
    if not simulate_ok:
        failures += 1

    surface_ok, surface_message = _check_surface_graph_model()
    print(surface_message)
    if not surface_ok:
        failures += 1

    operator_ok, operator_message = _check_operator_baseline()
    print(operator_message)
    if not operator_ok:
        failures += 1

    metrics_ok, metrics_message = _check_metrics_and_ood_hooks()
    print(metrics_message)
    if not metrics_ok:
        failures += 1

    sem_ok, sem_message = _check_sem_io_and_map_calibration()
    print(sem_message)
    if not sem_ok:
        failures += 1

    observation_ok, observation_message = _check_observation_projection_and_calibration()
    print(observation_message)
    if not observation_ok:
        failures += 1

    prior_ok, prior_message = _check_shape_prior_hooks()
    print(prior_message)
    if not prior_ok:
        failures += 1

    sbi_ok, sbi_message = _check_sbi_posterior_estimation()
    print(sbi_message)
    if not sbi_ok:
        failures += 1
    if full and sbi_message.startswith("skip(optional):"):
        failures += 1

    distill_ok, distill_message = _check_teacher_student_distillation()
    print(distill_message)
    if not distill_ok:
        failures += 1

    pipeline_ok, pipeline_message = _check_pipeline_stage_workflow(include_inference=not quick)
    print(pipeline_message)
    if not pipeline_ok:
        failures += 1

    advanced_ok, advanced_message = _check_pipeline_advanced_features()
    print(advanced_message)
    if not advanced_ok:
        failures += 1

    guard_ok, guard_message = _check_contract_and_split_guards()
    print(guard_message)
    if not guard_ok:
        failures += 1

    cache_ok, cache_message = _check_tracked_cache_artifacts()
    print(cache_message)
    if not cache_ok:
        failures += 1

    if quick:
        print("quick mode: optional dependency checks are informational only.")

    if full and capabilities_obj is not None and bool(capabilities_obj.minkowski_engine_shim):
        print("fail: verify --full requires a real MinkowskiEngine install; local compatibility shim is not allowed")
        failures += 1

    for module_name in optional_modules:
        ok, message = _check_import(module_name)
        if ok:
            print(message)
            continue
        if full:
            print(message)
            failures += 1
        else:
            print(message.replace("fail:", "skip(optional):", 1))

    if failures:
        print(f"verify: {failures} required check(s) failed.")
        return 1

    print("verify: all required checks passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wafer-surrogate-verify",
        description="Verification gate for local scaffold checks.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run minimal required checks and treat optional deps as informational.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Require optional dependencies and fail when optional backend checks cannot run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    if bool(args.quick) and bool(args.full):
        parser.error("--quick and --full cannot be used together")
    return run_verify(quick=bool(args.quick), full=bool(args.full))


if __name__ == "__main__":
    raise SystemExit(main())
