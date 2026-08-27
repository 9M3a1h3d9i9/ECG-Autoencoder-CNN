"""Command-line training entry point for preprocessed ECG beats."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.models import build_autoencoder, build_cnn_classifier, build_hybrid_classifier
from src.utils import ensure_dir, set_seed


def load_split(path: Path):
    data = np.load(path)
    return data["x"].astype("float32"), data["y"].astype("int64")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", choices=["cnn", "autoencoder", "hybrid"], default="cnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    set_seed(args.seed)
    x_train, y_train = load_split(args.data_dir / "train.npz")
    x_val, y_val = load_split(args.data_dir / "val.npz")
    if x_train.ndim == 2:
        x_train = x_train[..., None]
        x_val = x_val[..., None]

    output_dir = ensure_dir(args.output_dir / args.model)
    input_length = x_train.shape[1]
    num_classes = int(max(y_train.max(), y_val.max()) + 1)

    if args.model == "autoencoder":
        model = build_autoencoder(input_length)
        model.compile(optimizer="adam", loss="mse")
        train_target, val_target = x_train, x_val
        monitor = "val_loss"
    else:
        builder = build_cnn_classifier if args.model == "cnn" else build_hybrid_classifier
        model = builder(input_length, num_classes)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        train_target, val_target = y_train, y_val
        monitor = "val_loss"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output_dir / "best.keras", monitor=monitor, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=7, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(output_dir / "history.csv"),
    ]
    history = model.fit(x_train, train_target, validation_data=(x_val, val_target), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
    model.save(output_dir / "final.keras")
    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history.history, handle, indent=2)


if __name__ == "__main__":
    main()
