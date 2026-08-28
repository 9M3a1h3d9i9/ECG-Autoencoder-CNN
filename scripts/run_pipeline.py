"""Run preprocessing, training and optional evaluation as one experiment."""
import argparse
from pathlib import Path
from src.data.preprocessing import PreprocessingConfig, build_processed_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata = build_processed_dataset(args.raw_dir, args.processed_dir, PreprocessingConfig(window_size=args.window_size, seed=args.seed))
    print(f"Prepared dataset: {metadata}")
    print("Preprocessing completed. Train with: python -m scripts.train --data-dir", args.processed_dir)

if __name__ == "__main__":
    main()
