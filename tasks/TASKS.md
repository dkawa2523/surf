# TASKS

`tasks/tasks.json` がSSOTです。この文書は一覧用（生成）。

## P0

- P0-001: Initialize minimal Python package skeleton (src layout) + CLI + verify
- P0-002: Add modular registries: feature extractors, preprocess pipelines, models
- P0-003: Implement synthetic dataset generator (time-series SDF) for smoke runs
- P0-004: Implement minimal SDF/level-set geometry utilities
- P0-005: Implement baseline surrogate: predict Vn + rollout simulate
- P0-006: CLI wiring: train/infer/eval pipeline + runs/ outputs
- P0-007: Implement metrics + summary reporting + OOD hook (minimal)
- P0-008: Traceability + docs alignment (requirements->tasks)
- P0-009: Consistency sweep: commands, package name, docs, tasks
- P0-CHECKPOINT: Checkpoint: P0 completed
- P0-VIZ-001: Add VTI/PVD time-series export (SDF/fields) + CLI (viz export-vti)
- P0-VIZ-002: Add 3D quicklook visualization (slice PNGs) after SDF processing + CLI (viz render-slices)
- P0-VIZ-003: Add validation/test cross-section contour overlay (pred vs gt) + eval integration
- P0-VIZ-004: Add training/eval plots (loss/metrics curves + hist) + CLI (viz plot-metrics)
- P0-010: Implement batch inference + minimal recipe random search + pipeline combo sweep

## P1

- DECISION-P1-001: Decide constraints and dependency policy for P1/P2
- DECISION-P1-002: Decide YAML adoption policy and config compatibility
- P1-002: Define and implement HDF5/Zarr dataset schema for narrow-band SDF
- P1-003: Implement Vn pseudo-label generation from phi_t and phi_{t+1}
- P1-004: SEM feature I/O + latent z calibration (MAP)
- P1-005: Pipeline common types + manifest + stage output layout
- P1-006: Implement Data Cleaning stage
- P1-007: Implement Featurization stage with reconstruction bundle
- P1-008: Implement Preprocessing stage with inverse metadata
- P1-009: Implement Train stage with model comparison output
- P1-010: Implement Inference stage (single/batch/optimize) + OOD
- P1-011: Implement leaderboard aggregation (data_path/model_path)
- P1-012: CLI integration for pipeline command while keeping compatibility
- P1-013: Verification split and regression extension for pipeline
- P1-014: Remove unnecessary generated files and consolidate duplicated helpers
- DECISION-P1-003: Decide optimization engine policy (random/grid only vs optional external optimizer)

## P2

- P2-001: Teacher-Student distillation with privileged simulation logs (optional)
- P2-002: Surface graph model (non-local coupling) + SDF projection (optional)
- P2-003: Neural-operator baseline (phi0,c,t -> phi(t)) (optional)
- P2-004: SBI posterior estimation for latent z (optional)
- P2-005: Recipe search (BO/MFBO) (optional)
- P2-006: Add observation model interface stub (SDF→SEM feature projection) for future differentiable rendering
- P2-007: Add generative prior interface stub (shape prior) for future diffusion/score models
