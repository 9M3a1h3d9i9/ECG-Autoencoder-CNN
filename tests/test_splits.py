import pytest

from src.data.splits import split_records


def test_record_split_is_disjoint_and_complete():
    records = [str(i) for i in range(10)]
    split = split_records(records, seed=7)
    all_ids = set(split["train"] + split["val"] + split["test"])
    assert all_ids == set(records)
    assert not (set(split["train"]) & set(split["val"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["val"]) & set(split["test"]))


def test_split_requires_enough_records():
    with pytest.raises(ValueError):
        split_records(["100", "101"])
