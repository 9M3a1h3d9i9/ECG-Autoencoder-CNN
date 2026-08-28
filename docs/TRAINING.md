# Training and Experiment Layer

## Configuration

`configs/experiment.yaml` is the single source for the default experiment settings: seed, beat length, split ratios, batch size, epochs, learning rate, patience, and model-specific parameters.

## Class imbalance

Arrhythmia datasets are not uniformly distributed. Supervised training therefore exposes `compute_class_weights()` so the loss can give more importance to minority classes. Class weights must be computed from the training labels only; validation and test distributions must never influence training.

## Callbacks

The reusable training layer provides:

- `ModelCheckpoint` to retain the best validation model;
- `EarlyStopping` to limit overfitting and wasted computation;
- `ReduceLROnPlateau` to lower the learning rate when validation loss stops improving;
- `CSVLogger` for machine-readable experiment history.

## Curves

`plot_history()` writes loss and accuracy curves to the experiment output directory. Training curves are diagnostics, not substitutes for held-out test evaluation.

## Scientific protocol

The test set is used only after model selection. Hyperparameters and early stopping decisions must be based on training/validation data. Report class-wise precision, recall and F1 in addition to aggregate accuracy because accuracy can hide poor minority-class performance.
