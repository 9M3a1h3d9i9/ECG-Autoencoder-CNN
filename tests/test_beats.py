import numpy as np
import pytest

from src.data.beats import extract_beats, normalize_beat


def test_zscore_normalization_has_zero_mean():
    beat = np.array([1.0, 2.0, 3.0, 4.0])
    normalized = normalize_beat(beat)
    assert np.isclose(normalized.mean(), 0.0)


def test_extract_beats_skips_boundaries_and_maps_labels():
    signal = np.arange(100, dtype=float)
    beats, labels = extract_beats(
        signal,
        samples=[5, 50, 99],
        symbols=["N", "V", "N"],
        before=10,
        after=10,
    )
    assert beats.shape == (1, 20)
    assert labels.tolist() == ["V"]


def test_extract_beats_rejects_mismatched_annotations():
    with pytest.raises(ValueError):
        extract_beats(np.arange(20), [10], ["N", "V"])
