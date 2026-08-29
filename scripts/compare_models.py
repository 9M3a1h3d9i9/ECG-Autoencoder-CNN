"""Compare saved classifier metrics produced by evaluation runs."""
import argparse
import csv
import json
from pathlib import Path

from src.data.labels import AAMI_CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--models", nargs="+", default=["cnn", "hybrid"])
    args = parser.parse_args()
    rows = []
    for model_name in args.models:
        path = args.root / model_name / "evaluation" / "metrics.json"
        if not path.exists():
            path = args.root / "evaluation" / model_name / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"model": model_name, **{key: metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")}})
    if not rows:
        raise FileNotFoundError("No evaluation metrics found. Run scripts/evaluate.py first.")
    output = args.root / "model_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)

if __name__ == "__main__":
    main()
