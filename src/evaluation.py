"""Evaluation helpers for ECG classification experiments."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


def evaluate_classifier(y_true: Sequence[int], probabilities: np.ndarray, class_names: Sequence[str]) -> dict:
    """Return predictions, report and confusion matrix from class probabilities."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    y_pred = probabilities.argmax(axis=1)
    labels = list(range(len(class_names)))
    return {
        "predictions": y_pred,
        "report": classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def save_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str], output_path: str | Path) -> None:
    """Render and save a confusion matrix without requiring notebook code."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, colorbar=False, values_format="d")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
