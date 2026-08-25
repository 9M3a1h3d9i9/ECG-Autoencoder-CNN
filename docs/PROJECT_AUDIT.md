# ECG-Autoencoder-CNN Project Audit

## Current state

The repository contains a minimal WFDB loader and two exploratory notebooks with saved Keras models, but the reusable training and evaluation pipeline is not yet implemented.

## Key gaps

- No reproducible preprocessing pipeline.
- No explicit beat segmentation module.
- No train/validation/test split strategy at patient/record level.
- No reusable model implementations in `src`.
- No training or evaluation entry points.
- No automated tests.
- No experiment configuration.
- Documentation does not yet describe an end-to-end reproducible workflow.

## Target architecture

```text
src/
  data/        WFDB discovery, loading, preprocessing, beat extraction
  models/      Autoencoder and CNN implementations
  training/    Training loops and experiment utilities
  evaluation/  Metrics and reporting
  utils/       Reproducibility and shared helpers
configs/       Experiment configuration
scripts/       Command-line entry points
tests/         Unit tests
docs/          Design and usage documentation
```

## Development principle

All changes are developed on `feature/complete-research-pipeline`. The `main` branch remains untouched until review and validation are complete.

## First implementation milestone

Build a deterministic data foundation: record discovery, annotation handling, beat extraction, preprocessing, class mapping, dataset splitting, and tests. Model training will be built on top of this validated foundation.
