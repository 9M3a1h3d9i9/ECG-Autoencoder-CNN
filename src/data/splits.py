from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def split_records(
    record_ids: Sequence[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split at record level to avoid mixing beats from one record across splits."""
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("ratios must be within [0, 1]")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    ids = list(dict.fromkeys(record_ids))
    if len(ids) < 3:
        raise ValueError("at least three unique records are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    train_end = max(1, int(len(ids) * train_ratio))
    val_end = train_end + max(1, int(len(ids) * val_ratio))
    if val_end >= len(ids):
        val_end = len(ids) - 1
    return {
        "train": sorted(ids[:train_end]),
        "val": sorted(ids[train_end:val_end]),
        "test": sorted(ids[val_end:]),
    }
