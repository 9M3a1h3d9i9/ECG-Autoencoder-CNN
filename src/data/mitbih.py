from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import wfdb


@dataclass(frozen=True)
class ECGRecord:
    """A validated MIT-BIH record and its annotation prefix."""

    name: str
    path: Path


class MITBIHRecordLoader:
    """Discover and load local MIT-BIH Arrhythmia Database records.

    The loader keeps filesystem logic separate from preprocessing so that
    experiments can be reproduced from an explicit list of record IDs.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()

    def validate(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.data_dir}")
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {self.data_dir}")

    def list_records(self) -> list[str]:
        """Return sorted record IDs that contain both signal and header files."""
        self.validate()
        records = []
        for hea in self.data_dir.glob("*.hea"):
            stem = hea.stem
            if (self.data_dir / f"{stem}.dat").exists():
                records.append(stem)
        return sorted(records)

    def records(self, names: Iterable[str] | None = None) -> list[ECGRecord]:
        available = set(self.list_records())
        selected = sorted(available if names is None else set(names))
        missing = [name for name in selected if name not in available]
        if missing:
            raise FileNotFoundError(f"Missing MIT-BIH records: {', '.join(missing)}")
        return [ECGRecord(name=name, path=self.data_dir / name) for name in selected]

    def load_record(self, record_name: str):
        return wfdb.rdrecord(str(self.data_dir / record_name))

    def load_annotation(self, record_name: str, extension: str = "atr"):
        annotation_path = self.data_dir / f"{record_name}.{extension}"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation not found: {annotation_path}")
        return wfdb.rdann(str(self.data_dir / record_name), extension)
