"""Backward-compatible dataset entry point.

New code should import from :mod:`src.data`.
"""

from pathlib import Path

from src.data.mitbih import MITBIHRecordLoader


class ECGDataset(MITBIHRecordLoader):
    """Compatibility wrapper around :class:`MITBIHRecordLoader`."""

    def __init__(self, data_dir: str | Path) -> None:
        super().__init__(data_dir)
