"""Shared reproducibility and filesystem helpers."""

import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy and TensorFlow seeds for repeatable experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
