"""Reusable training utilities for ECG classification."""

from pathlib import Path

import numpy as np
import tensorflow as tf

from .utils import ensure_dir


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute balanced weights without requiring imbalanced-learn."""
    y = np.asarray(y, dtype=np.int64)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) == 0:
        raise ValueError("Cannot compute class weights for an empty label array")
    total = counts.sum()
    return {int(cls): float(total / (len(classes) * count)) for cls, count in zip(classes, counts)}


def make_callbacks(output_dir: str | Path, monitor: str = "val_loss", patience: int = 7):
    """Create standard callbacks shared by all supervised experiments."""
    output_dir = ensure_dir(output_dir)
    return [
        tf.keras.callbacks.ModelCheckpoint(output_dir / "best.keras", monitor=monitor, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(output_dir / "history.csv"),
    ]


def plot_history(history, output_path: str | Path) -> None:
    """Save loss and available accuracy curves as separate figures."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["loss"]
    if "accuracy" in history.history:
        metrics.append("accuracy")
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(history.history[metric], label=f"train {metric}")
        validation_key = f"val_{metric}"
        if validation_key in history.history:
            ax.plot(history.history[validation_key], label=f"validation {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.title())
        ax.legend()
        fig.tight_layout()
        suffix = metric.replace("/", "_")
        fig.savefig(output_path.with_name(f"{output_path.stem}_{suffix}.png"), dpi=150)
        plt.close(fig)
