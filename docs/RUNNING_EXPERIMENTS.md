# Running ECG Experiments

## 1. Prepare data

Place the MIT-BIH record files under `data/raw/`. The data layer is responsible for discovering WFDB records, loading annotations, extracting fixed-length beats and mapping beat symbols to AAMI groups.

Processed splits should be stored as:

- `data/processed/train.npz`
- `data/processed/val.npz`
- `data/processed/test.npz`

Each archive contains `x` with shape `(n_samples, signal_length)` and `y` with integer class IDs.

## 2. Train a baseline

```bash
python -m scripts.train --model cnn --epochs 30
```

## 3. Train the autoencoder

```bash
python -m scripts.train --model autoencoder --epochs 30
```

## 4. Train the hybrid model

```bash
python -m scripts.train --model hybrid --epochs 30
```

## 5. Evaluate

```bash
python -m scripts.evaluate --model outputs/cnn/best.keras
```

## Reproducibility

Use a fixed seed and keep record-level splits stable. Do not create beat-level random splits from the complete dataset because beats from the same ECG record can otherwise appear in both training and evaluation data.
