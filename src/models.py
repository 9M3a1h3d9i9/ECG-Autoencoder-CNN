"""TensorFlow/Keras models for ECG beat experiments."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_classifier(input_length: int, num_classes: int, dropout: float = 0.2) -> keras.Model:
    """Build a compact 1D CNN classifier for fixed-length ECG beats."""
    inputs = keras.Input(shape=(input_length, 1), name="ecg")
    x = inputs
    for filters in (32, 64, 128):
        x = layers.Conv1D(filters, 5, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class")(x)
    return keras.Model(inputs, outputs, name="ecg_cnn")


def build_autoencoder(input_length: int, latent_channels: int = 32) -> keras.Model:
    """Build a convolutional ECG autoencoder with reconstruction output."""
    inputs = keras.Input(shape=(input_length, 1), name="ecg")
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(inputs)
    x = layers.MaxPool1D(2, padding="same")(x)
    x = layers.Conv1D(latent_channels, 5, padding="same", activation="relu", name="latent")(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    outputs = layers.Conv1D(1, 3, padding="same", activation="linear", name="reconstruction")(x)
    return keras.Model(inputs, outputs, name="ecg_autoencoder")


def build_hybrid_classifier(input_length: int, num_classes: int, latent_channels: int = 32, dropout: float = 0.2) -> keras.Model:
    """Build an end-to-end encoder plus classifier model."""
    inputs = keras.Input(shape=(input_length, 1), name="ecg")
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(inputs)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(latent_channels, 5, padding="same", activation="relu")(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class")(x)
    return keras.Model(inputs, outputs, name="ecg_hybrid")
