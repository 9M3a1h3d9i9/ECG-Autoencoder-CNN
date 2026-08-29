"""Evaluate a trained classifier on the untouched test split."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from src.data.labels import AAMI_CLASSES
from src.data.validation import validate_arrays
from src.evaluation import evaluate_classifier, save_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/test.npz"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    data = np.load(args.data)
    x, y = data["x"].astype("float32"), data["y"].astype("int64")
    validate_arrays(x, y, expected_length=x.shape[1], num_classes=len(AAMI_CLASSES))
    if x.ndim == 2:
        x = x[..., None]
    model = tf.keras.models.load_model(args.model)
    probabilities = model.predict(x, verbose=0)
    result = evaluate_classifier(y, probabilities, AAMI_CLASSES)
    save_evaluation(result, AAMI_CLASSES, args.output_dir)
    print(json.dumps({key: value for key, value in result.items() if isinstance(value, float)}, indent=2))

if __name__ == "__main__":
    main()
