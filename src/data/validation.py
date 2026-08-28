"""Validation checks for processed ECG datasets."""

import numpy as np


def validate_arrays(x: np.ndarray, y: np.ndarray, expected_length: int = 256, num_classes: int = 5) -> None:
    """Raise a descriptive error when processed arrays violate dataset invariants."""
    if x.ndim not in (2, 3):
        raise ValueError(f"x must have 2 or 3 dimensions; got {x.shape}")
    if x.shape[0] != len(y):
        raise ValueError("x and y contain different numbers of samples")
    if x.shape[1] != expected_length:
        raise ValueError(f"Expected beat length {expected_length}; got {x.shape[1]}")
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or infinite values")
    if len(y) and (y.min() < 0 or y.max() >= num_classes):
        raise ValueError("y contains an invalid class ID")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y must contain integer class IDs")


def validate_split_disjointness(*sources: np.ndarray) -> None:
    """Ensure record identifiers do not overlap across dataset splits."""
    sets = [set(values.tolist()) for values in sources]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise ValueError(f"Record leakage detected between splits: {sorted(overlap)}")
