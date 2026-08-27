"""Evaluate a trained ECG classifier on a saved test split."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.evaluation import evaluate_classifier, save_confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/test.npz"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--class-names", nargs="+", default=["N", "S", "V", "F", "Q"])
    args = parser.parse_args()

    data = np.load(args.data)
    x, y = data["x"].astype("float32"), data["y"].astype("int64")
    if x.ndim == 2:
        x = x[..., None]
    model = tf.keras.models.load_model(args.model)
    probabilities = model.predict(x, verbose=0)
    result = evaluate_classifier(y, probabilities, args.class_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(result["confusion_matrix"], args.class_names, args.output_dir / "confusion_matrix.png")
    with open(args.output_dir / "classification_report.json", "w", encoding="utf-8") as handle:
        json.dump(result["report"], handle, indent=2)
    print(json.dumps(result["report"].get("accuracy", {}), indent=2))


if __name__ == "__main__":
    main()
