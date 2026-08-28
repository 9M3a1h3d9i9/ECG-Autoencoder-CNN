"""Train reproducible ECG experiments from processed NPZ splits."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from src.data.validation import validate_arrays
from src.models import build_autoencoder, build_cnn_classifier, build_hybrid_classifier
from src.training import compute_class_weights, make_callbacks, plot_history
from src.utils import ensure_dir, set_seed


def load_split(path):
    data = np.load(path)
    return data["x"].astype("float32"), data["y"].astype("int64")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", choices=["cnn", "autoencoder", "hybrid"], default="cnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    set_seed(args.seed)
    x_train, y_train = load_split(args.data_dir / "train.npz")
    x_val, y_val = load_split(args.data_dir / "val.npz")
    validate_arrays(x_train, y_train, expected_length=x_train.shape[1])
    validate_arrays(x_val, y_val, expected_length=x_train.shape[1])
    if x_train.ndim == 2:
        x_train, x_val = x_train[..., None], x_val[..., None]
    output_dir = ensure_dir(args.output_dir / args.model)
    input_length = x_train.shape[1]
    num_classes = 5
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    if args.model == "autoencoder":
        model = build_autoencoder(input_length)
        model.compile(optimizer=optimizer, loss="mse")
        train_target, val_target, class_weight = x_train, x_val, None
    else:
        builder = build_cnn_classifier if args.model == "cnn" else build_hybrid_classifier
        model = builder(input_length, num_classes)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        train_target, val_target = y_train, y_val
        class_weight = compute_class_weights(y_train)
    history = model.fit(x_train, train_target, validation_data=(x_val, val_target), epochs=args.epochs, batch_size=args.batch_size, class_weight=class_weight, callbacks=make_callbacks(output_dir, patience=args.patience))
    model.save(output_dir / "final.keras")
    plot_history(history, output_dir / "history.png")
    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history.history, handle, indent=2)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(args.output_dir)}, handle, default=str, indent=2)

if __name__ == "__main__":
    main()
