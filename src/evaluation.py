"""Evaluation helpers for reproducible ECG classification experiments."""
from pathlib import Path
from typing import Sequence
import json
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_classifier(y_true: Sequence[int], probabilities: np.ndarray, class_names: Sequence[str]) -> dict:
    """Compute global and class-wise classification metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError("probabilities must have shape (n_samples, len(class_names))")
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same number of samples")
    y_pred = probabilities.argmax(axis=1)
    labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "class_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_evaluation(result: dict, class_names: Sequence[str], output_dir: str | Path) -> None:
    """Save JSON metrics and confusion-matrix visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({**result, "class_names": list(class_names)}, handle, indent=2)
    save_confusion_matrix(np.asarray(result["confusion_matrix"]), class_names, output_dir / "confusion_matrix.png")


def save_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str], output_path: str | Path) -> None:
    """Render and save a confusion matrix."""
    import matplotlib.pyplot as plt
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, colorbar=False, values_format="d")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
