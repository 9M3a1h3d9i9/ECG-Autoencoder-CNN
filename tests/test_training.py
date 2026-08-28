import numpy as np

from src.training import compute_class_weights


def test_class_weights_balance_inverse_frequency():
    weights = compute_class_weights(np.array([0, 0, 0, 1, 1, 2]))
    assert weights[0] < weights[1] < weights[2]


def test_class_weights_reject_empty_labels():
    try:
        compute_class_weights(np.array([], dtype=np.int64))
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
