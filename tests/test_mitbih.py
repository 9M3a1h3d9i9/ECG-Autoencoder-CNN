import pytest

from src.data.mitbih import MITBIHRecordLoader


def test_missing_dataset_directory_is_reported(tmp_path):
    loader = MITBIHRecordLoader(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        loader.list_records()


def test_record_discovery_requires_header_and_signal(tmp_path):
    (tmp_path / "100.hea").write_text("header")
    (tmp_path / "101.dat").write_bytes(b"data")
    (tmp_path / "101.hea").write_text("header")
    loader = MITBIHRecordLoader(tmp_path)
    assert loader.list_records() == ["101"]


def test_missing_selected_record_is_reported(tmp_path):
    (tmp_path / "100.dat").write_bytes(b"data")
    (tmp_path / "100.hea").write_text("header")
    with pytest.raises(FileNotFoundError):
        MITBIHRecordLoader(tmp_path).records(["999"])
