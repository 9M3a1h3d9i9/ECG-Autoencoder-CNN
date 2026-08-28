# ECG Data Pipeline

The project converts raw MIT-BIH WFDB records into deterministic, record-level train, validation and test datasets.

## Input

Place WFDB record files and their annotation files in `data/raw/`.

## Processing steps

1. Discover records.
2. Split records before beat extraction.
3. Load signal and annotation data.
4. Extract fixed-length windows centered on annotated beats.
5. Drop windows that cross signal boundaries.
6. Normalize each extracted beat when enabled.
7. Map beat symbols to AAMI groups.
8. Encode class names as integer IDs.
9. Save compressed NPZ archives with source-record metadata.

Splitting before aggregation is intentional: it prevents beats from the same ECG record from leaking into both training and evaluation sets.

## Run

```bash
python -m scripts.preprocess --raw-dir data/raw --output-dir data/processed
```

The command writes:

- `train.npz`
- `val.npz`
- `test.npz`
- `metadata.json`

Each NPZ archive contains:

- `x`: ECG beats.
- `y`: integer class IDs.
- `source`: originating record IDs.

`metadata.json` records sample counts, record counts and class distributions for experiment auditing.
