from __future__ import annotations

from typing import Sequence

import numpy as np


AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q", "?": "Q",
}


def normalize_beat(beat: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize one beat without using statistics from other records."""
    beat = np.asarray(beat, dtype=np.float32)
    if method == "none":
        return beat
    if method == "zscore":
        std = float(beat.std())
        return (beat - beat.mean()) / (std if std > 1e-8 else 1.0)
    if method == "minmax":
        low, high = float(beat.min()), float(beat.max())
        scale = high - low
        return (beat - low) / (scale if scale > 1e-8 else 1.0)
    raise ValueError(f"Unsupported normalization method: {method}")


def extract_beats(
    signal: np.ndarray,
    samples: Sequence[int],
    symbols: Sequence[str],
    before: int = 90,
    after: int = 90,
    normalization: str = "zscore",
    allowed_symbols: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed-size annotated beats from a one-dimensional ECG signal.

    Beats touching a record boundary are skipped instead of padded, preventing
    artificial values from entering the training set.
    """
    signal = np.asarray(signal).squeeze()
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if len(samples) != len(symbols):
        raise ValueError("samples and symbols must have equal length")

    beats, labels = [], []
    allowed = allowed_symbols if allowed_symbols is not None else set(AAMI_MAP)
    for sample, symbol in zip(samples, symbols):
        if symbol not in allowed or symbol not in AAMI_MAP:
            continue
        start, end = int(sample) - before, int(sample) + after
        if start < 0 or end > len(signal):
            continue
        beat = normalize_beat(signal[start:end], normalization)
        beats.append(beat)
        labels.append(AAMI_MAP[symbol])

    length = before + after
    if not beats:
        return np.empty((0, length), dtype=np.float32), np.empty((0,), dtype=str)
    return np.stack(beats).astype(np.float32), np.asarray(labels)
