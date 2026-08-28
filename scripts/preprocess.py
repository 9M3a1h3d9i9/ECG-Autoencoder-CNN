"""Build processed ECG train/validation/test splits from raw MIT-BIH records."""

import argparse
import json
from pathlib import Path

from src.data.preprocessing import PreprocessingConfig, build_processed_dataset


def main():
    parser = argparse.ArgumentParser(description="Preprocess MIT-BIH ECG records")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    config = PreprocessingConfig(
        window_size=args.window_size,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    metadata = build_processed_dataset(args.raw_dir, args.output_dir, config)
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
