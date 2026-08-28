import numpy as np
import pytest

from src.data.labels import CLASS_TO_ID, encode_aami, map_symbol
from src.data.validation import validate_arrays, validate_split_disjointness


def test_global_mapping_is_stable():
    assert CLASS_TO_ID == {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}
    assert encode_aami(["N", "V", "S"]) == [0, 2, 1]


def test_symbol_mapping():
    assert map_symbol("L") == "N"
    assert map_symbol("V") == "V"
    assert map_symbol("?") is None


def test_validation_accepts_valid_arrays():
    x = np.zeros((4, 256), dtype=np.float32)
    y = np.array([0, 1, 2, 4], dtype=np.int64)
    validate_arrays(x, y)


def test_validation_rejects_nan():
    x = np.zeros((2, 256), dtype=np.float32)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_arrays(x, np.array([0, 1]))


def test_validation_rejects_record_overlap():
    with pytest.raises(ValueError, match="leakage"):
        validate_split_disjointness(np.array(["100", "101"]), np.array(["101", "102"]))
