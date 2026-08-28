import numpy as np
import pytest
from src.data.validation import validate_arrays, validate_split_disjointness
from src.training import compute_class_weights


def test_processed_contract_and_training_contract():
    x = np.random.default_rng(42).normal(size=(12, 256)).astype(np.float32)
    y = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4, 4], dtype=np.int64)
    validate_arrays(x, y)
    weights = compute_class_weights(y)
    assert set(weights) == {0, 1, 2, 3, 4}


def test_split_contract_rejects_leakage():
    with pytest.raises(ValueError):
        validate_split_disjointness(np.array(["100"]), np.array(["101"]), np.array(["100"]))
