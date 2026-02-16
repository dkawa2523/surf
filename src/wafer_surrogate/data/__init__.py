__all__ = [
    "BackendUnavailableError",
    "DatasetSchemaError",
    "InMemoryAdapter",
    "NarrowBandDataset",
    "NarrowBandRun",
    "NarrowBandStep",
    "read_hdf5_dataset",
    "read_narrow_band_dataset",
    "read_sem_features_csv",
    "read_sem_features_json",
    "read_zarr_dataset",
    "SemFeatureError",
    "SemFeatureVector",
    "SyntheticSDFDataset",
    "SyntheticSDFRun",
    "generate_synthetic_sdf_dataset",
    "load_sem_features",
    "load_narrow_band_dataset",
    "NarrowBandDatasetReader",
    "PointSampler",
    "SplitPolicy",
    "split_runs",
    "PRIV_FEATURE_NAMES",
    "dense_priv_matrix",
    "generate_proxy_privileged_lookup",
    "load_privileged_logs_h5",
    "load_privileged_logs_jsonl",
    "resolve_privileged_lookup",
    "synthetic_to_narrow_band_dataset",
    "write_hdf5_dataset",
    "write_narrow_band_dataset",
    "write_synthetic_example",
    "write_zarr_dataset",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        from importlib import import_module

        synthetic = import_module("wafer_surrogate.data.synthetic")
        if hasattr(synthetic, name):
            return getattr(synthetic, name)
        io = import_module("wafer_surrogate.data.io")
        if hasattr(io, name):
            return getattr(io, name)
        sem = import_module("wafer_surrogate.data.sem")
        if hasattr(sem, name):
            return getattr(sem, name)
        mc_logs = import_module("wafer_surrogate.data.mc_logs")
        if hasattr(mc_logs, name):
            return getattr(mc_logs, name)
        h5_dataset = import_module("wafer_surrogate.data.h5_dataset")
        if hasattr(h5_dataset, name):
            return getattr(h5_dataset, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
