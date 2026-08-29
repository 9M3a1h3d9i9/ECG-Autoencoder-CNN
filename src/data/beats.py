from __future__ import annotations

from typing import Sequence
import numpy as np


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


def extract_beats(signal: np.ndarray, samples: Sequence[int], symbols: Sequence[str], before: int = 128, after: int = 128, normalization: str = "zscore", allowed_symbols: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed-length annotated beats; boundary beats are skipped."""
    signal = np.asarray(signal).squeeze()
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if len(samples) != len(symbols):
        raise ValueError("samples and symbols must have equal length")
    beats, labels = [], []
    allowed = allowed_symbols if allowed_symbols is not None else set(_AAMI_MAP)
    for sample, symbol in zip(samples, symbols):
        if symbol not in allowed:
            continue
        start, end = int(sample) - before, int(sample) + after
        if start < 0 or end > len(signal):
            continue
        beats.append(normalize_beat(signal[start:end], normalization))
        labels.append(_AAMI_MAP[symbol])
    length = before + after
    if not beats:
        return np.empty((0, length), dtype=np.float32), np.empty((0,), dtype="U1")
    return np.stack(beats).astype(np.float32), np.asarray(labels)


_AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V", "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
