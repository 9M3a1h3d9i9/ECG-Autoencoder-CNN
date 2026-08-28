"""End-to-end preprocessing for annotated MIT-BIH ECG records."""

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np

from .beats import extract_beats
from .labels import AAMI_CLASSES, CLASS_TO_ID, map_symbol
from .mitbih import discover_records, load_record
from .splits import split_records
from .validation import validate_arrays, validate_split_disjointness


@dataclass(frozen=True)
class PreprocessingConfig:
    window_size: int = 256
    normalize: bool = True
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15


def process_records(record_paths: Iterable[Path], config: PreprocessingConfig):
    signals, labels, sources = [], [], []
    for record_path in record_paths:
        record, annotation = load_record(record_path)
        beats, symbols = extract_beats(record.p_signal, annotation.sample, annotation.symbol, window_size=config.window_size, normalize=config.normalize)
        mapped = [map_symbol(symbol) for symbol in symbols]
        valid = [i for i, label in enumerate(mapped) if label is not None]
        if valid:
            signals.append(beats[valid])
            labels.extend(mapped[i] for i in valid)
            sources.extend([record_path.stem] * len(valid))
    if not signals:
        raise RuntimeError("No valid ECG beats were extracted.")
    return np.concatenate(signals).astype("float32"), np.asarray(labels), np.asarray(sources)


def build_processed_dataset(raw_dir: str | Path, output_dir: str | Path, config: PreprocessingConfig = PreprocessingConfig()):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    records = discover_records(raw_dir)
    if not records:
        raise FileNotFoundError(f"No WFDB records found in {raw_dir}")
    split = split_records(records, train_ratio=config.train_ratio, val_ratio=config.val_ratio, seed=config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed, metadata = {}, {}
    for name, record_paths in split.items():
        x, raw_labels, sources = process_records(record_paths, config)
        y = np.asarray([CLASS_TO_ID[label] for label in raw_labels], dtype=np.int64)
        validate_arrays(x, y, expected_length=config.window_size, num_classes=len(AAMI_CLASSES))
        processed[name] = sources
        np.savez_compressed(output_dir / f"{name}.npz", x=x, y=y, source=sources)
        metadata[name] = {"samples": int(len(x)), "records": int(len(set(sources.tolist()))), "class_counts": dict(Counter(raw_labels.tolist()))}
    validate_split_disjointness(processed["train"], processed["val"], processed["test"])
    metadata["class_names"] = list(AAMI_CLASSES)
    metadata["class_to_id"] = CLASS_TO_ID
    metadata["config"] = asdict(config)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata
