import numpy as np

from src.data.preprocessing import encode_labels


def test_encode_labels_is_deterministic():
    labels = np.asarray(["V", "N", "S", "N"])
    encoded, classes = encode_labels(labels)
    assert classes == ["N", "S", "V"]
    assert encoded.tolist() == [2, 0, 1, 0]


def test_encoded_labels_are_contiguous():
    labels = np.asarray(["F", "Q", "F"])
    encoded, classes = encode_labels(labels)
    assert set(encoded.tolist()) == set(range(len(classes)))
