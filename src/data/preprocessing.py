"""End-to-end preprocessing for annotated MIT-BIH ECG records."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .beats import extract_beats, map_symbols_to_aami
from .mitbih import discover_records, load_record
from .splits import split_records


@dataclass(frozen=True)
class PreprocessingConfig:
    window_size: int = 256
    normalize: bool = True
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15


def process_records(record_paths: Iterable[Path], config: PreprocessingConfig):
    """Extract normalized beats and AAMI labels from a collection of records."""
    signals, labels, sources = [], [], []
    for record_path in record_paths:
        record, annotation = load_record(record_path)
        beats, symbols = extract_beats(
            record.p_signal,
            annotation.sample,
            annotation.symbol,
            window_size=config.window_size,
            normalize=config.normalize,
        )
        mapped = map_symbols_to_aami(symbols)
        valid = [index for index, label in enumerate(mapped) if label is not None]
        if valid:
            signals.append(beats[valid])
            labels.extend(mapped[index] for index in valid)
            sources.extend([record_path.stem] * len(valid))
    if not signals:
        raise RuntimeError("No valid ECG beats were extracted.")
    return np.concatenate(signals, axis=0).astype("float32"), np.asarray(labels), np.asarray(sources)


def encode_labels(labels: np.ndarray):
    """Encode sorted AAMI labels as integer IDs and return the class mapping."""
    class_names = sorted(set(labels.tolist()))
    mapping = {name: index for index, name in enumerate(class_names)}
    return np.asarray([mapping[label] for label in labels], dtype=np.int64), class_names


def build_processed_dataset(raw_dir: str | Path, output_dir: str | Path, config: PreprocessingConfig = PreprocessingConfig()):
    """Create deterministic train/validation/test NPZ files from raw MIT-BIH data."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    records = discover_records(raw_dir)
    if not records:
        raise FileNotFoundError(f"No WFDB records found in {raw_dir}")
    split = split_records(records, train_ratio=config.train_ratio, val_ratio=config.val_ratio, seed=config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, record_paths in split.items():
        x, raw_labels, sources = process_records(record_paths, config)
        y, class_names = encode_labels(raw_labels)
        np.savez_compressed(output_dir / f"{name}.npz", x=x, y=y, source=sources)
        metadata[name] = {
            "samples": int(len(x)),
            "records": int(len(set(sources.tolist()))),
            "class_names": class_names,
            "class_counts": dict(Counter(raw_labels.tolist())),
        }
    return metadata
