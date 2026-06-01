"""
Deep learning models: CNN, LSTM, CNN-LSTM-Attention (Das et al. 2025).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import DL_BATCH_SIZE, DL_EPOCHS, DL_LEARNING_RATE, RANDOM_STATE

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    HAS_TF = True
except ImportError:
    HAS_TF = False


def _require_tf() -> None:
    if not HAS_TF:
        raise ImportError(
            "TensorFlow is required for deep learning models. "
            "Install with: pip install tensorflow"
        )


def configure_tf_gpu(log: bool = True) -> list:
    """
    Enable GPU memory growth and return visible GPU devices.

    TensorFlow/Keras uses GPU automatically when available; this avoids OOM on
    Paperspace by not allocating all VRAM at once.
    """
    _require_tf()
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    if log:
        if gpus:
            print(f"TensorFlow GPU enabled: {len(gpus)} device(s) — {gpus}")
        else:
            print("TensorFlow: no GPU found — training will use CPU.")
    return gpus


def set_reproducibility(seed: int = RANDOM_STATE) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    if HAS_TF:
        tf.random.set_seed(seed)
        configure_tf_gpu(log=True)


class AttentionLayer(layers.Layer if HAS_TF else object):
    """Attention over LSTM sequence output."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (batch, time, features)
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(x * weights, axis=1)
        return context

    def get_config(self):
        return super().get_config()


def build_cnn_model(input_shape: tuple[int, int], num_classes: int) -> Any:
    """1D CNN classifier."""
    _require_tf()
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(32, 7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=DL_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm_model(input_shape: tuple[int, int], num_classes: int) -> Any:
    """Stacked LSTM classifier."""
    _require_tf()
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=DL_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_lstm_attention_model(
    input_shape: tuple[int, int], num_classes: int
) -> Any:
    """Hybrid CNN-LSTM-Attention model (main article architecture)."""
    _require_tf()
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(32, 7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    # Attention context vector
    score = layers.Dense(1, activation="tanh")(x)
    weights = layers.Softmax(axis=1)(score)
    context = layers.Multiply()([x, weights])
    context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(context)
    x = layers.Dense(128, activation="relu", name="embedding_dense")(context)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="cnn_lstm_attention")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=DL_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_deep_model(
    model_name: str,
    input_shape: tuple[int, int],
    num_classes: int,
) -> Any:
    """Factory for deep models."""
    builders = {
        "cnn": build_cnn_model,
        "lstm": build_lstm_model,
        "cnn_lstm_attention": build_cnn_lstm_attention_model,
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(builders)}")
    return builders[model_name](input_shape, num_classes)


def train_deep_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = DL_EPOCHS,
    batch_size: int = DL_BATCH_SIZE,
    model_name: str = "cnn_lstm_attention",
    output_dir: str | Path | None = None,
) -> tuple[Any, Any, float]:
    """
    Train deep model with early stopping and LR scheduling.

    Returns trained model, history, training time (seconds).
    """
    _require_tf()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(out / f"{model_name}_best.keras"),
                monitor="val_loss",
                save_best_only=True,
            )
        )

    t0 = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.perf_counter() - t0
    return model, history.history, train_time


def extract_deep_embeddings(
    model: Any,
    X: np.ndarray,
    layer_name: str = "embedding_dense",
) -> np.ndarray:
    """Extract penultimate layer embeddings for t-SNE visualization."""
    _require_tf()
    try:
        emb_model = keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output,
        )
    except ValueError:
        # Fallback to pre-softmax layer
        emb_model = keras.Model(
            inputs=model.input,
            outputs=model.layers[-2].output,
        )
    return emb_model.predict(X, verbose=0)
