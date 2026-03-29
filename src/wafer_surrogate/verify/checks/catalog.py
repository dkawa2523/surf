from __future__ import annotations

REQUIRED_MODULES = [
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

OPTIONAL_MODULES = [
    "numpy",
    "matplotlib",
    "torch",
    "MinkowskiEngine",
    "sbi",
    "botorch",
]
