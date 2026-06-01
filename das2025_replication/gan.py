"""
Conditional WGAN-GP for EEG data augmentation (Das et al. 2025).

GAN is trained ONLY on training data. Never use validation/test for GAN training.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .config import GAN_BATCH_SIZE, GAN_EPOCHS, GAN_LATENT_DIM, RANDOM_STATE

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    HAS_TF = True
except ImportError:
    HAS_TF = False


def _require_tf() -> None:
    if not HAS_TF:
        raise ImportError("TensorFlow required for GAN. pip install tensorflow")


def build_eeg_generator(
    latent_dim: int,
    output_shape: tuple[int, int],
    num_classes: int,
) -> Any:
    """
    Conditional generator: noise + class label -> EEG segment.

    output_shape: (n_samples, n_channels)
    """
    _require_tf()
    n_samples, n_channels = output_shape
    noise_in = keras.Input(shape=(latent_dim,))
    label_in = keras.Input(shape=(1,), dtype="int32")
    label_emb = layers.Embedding(num_classes, 50)(label_in)
    label_emb = layers.Flatten()(label_emb)
    merged = layers.Concatenate()([noise_in, label_emb])
    x = layers.Dense(256, activation="relu")(merged)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(n_samples * n_channels, activation="tanh")(x)
    out = layers.Reshape((n_samples, n_channels))(x)
    return keras.Model([noise_in, label_in], out, name="generator")


def build_eeg_discriminator(
    input_shape: tuple[int, int],
    num_classes: int,
) -> Any:
    """Conditional discriminator (critic) for WGAN-GP."""
    _require_tf()
    data_in = keras.Input(shape=input_shape)
    label_in = keras.Input(shape=(1,), dtype="int32")
    label_emb = layers.Embedding(num_classes, 50)(label_in)
    label_emb = layers.Flatten()(label_emb)
    label_emb = layers.RepeatVector(input_shape[0])(label_emb)
    label_emb = layers.Reshape((input_shape[0], 50))(label_emb)
    merged = layers.Concatenate(axis=-1)([data_in, label_emb])
    x = layers.Conv1D(64, 5, strides=2, padding="same")(merged)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1D(128, 5, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(1)(x)  # WGAN: no sigmoid
    return keras.Model([data_in, label_in], out, name="discriminator")


def _gradient_penalty(
    critic: Any,
    real: np.ndarray,
    fake: np.ndarray,
    labels: np.ndarray,
    gp_weight: float = 10.0,
) -> tf.Tensor:
    """Wasserstein gradient penalty."""
    batch_size = tf.shape(real)[0]
    alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
    interpolated = alpha * real + (1 - alpha) * fake
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic([interpolated, labels], training=True)
    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-8)
    return gp_weight * tf.reduce_mean((norm - 1.0) ** 2)


@tf.function
def _train_critic_step(
    critic, generator, real_x, labels, latent_dim, n_critic, gp_weight, opt_c
):
    batch_size = tf.shape(real_x)[0]
    total_c_loss = 0.0
    for _ in range(n_critic):
        noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape() as tape:
            fake_x = generator([noise, labels], training=True)
            real_score = critic([real_x, labels], training=True)
            fake_score = critic([fake_x, labels], training=True)
            gp = _gradient_penalty(critic, real_x, fake_x, labels, gp_weight)
            c_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + gp
        grads = tape.gradient(c_loss, critic.trainable_variables)
        opt_c.apply_gradients(zip(grads, critic.trainable_variables))
        total_c_loss += c_loss
    return total_c_loss / n_critic


@tf.function
def _train_generator_step(critic, generator, labels, latent_dim, batch_size, opt_g):
    noise = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape() as tape:
        fake_x = generator([noise, labels], training=True)
        g_loss = -tf.reduce_mean(critic([fake_x, labels], training=True))
    grads = tape.gradient(g_loss, generator.trainable_variables)
    opt_g.apply_gradients(zip(grads, generator.trainable_variables))
    return g_loss


def train_wgan_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    latent_dim: int = GAN_LATENT_DIM,
    epochs: int = GAN_EPOCHS,
    batch_size: int = GAN_BATCH_SIZE,
    n_critic: int = 5,
    gp_weight: float = 10.0,
    learning_rate: float = 1e-4,
) -> tuple[Any, Any, dict]:
    """
    Train conditional WGAN-GP on training data only.

    X_train shape: (n_trials, n_samples, n_channels)
    """
    _require_tf()
    tf.random.set_seed(RANDOM_STATE)
    output_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = int(np.max(y_train) + 1)

    generator = build_eeg_generator(latent_dim, output_shape, num_classes)
    critic = build_eeg_discriminator(output_shape, num_classes)

    opt_g = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.9)
    opt_c = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.9)

    history = {"c_loss": [], "g_loss": []}
    n_samples = len(X_train)
    steps_per_epoch = max(n_samples // batch_size, 1)

    for epoch in range(epochs):
        idx = np.random.permutation(n_samples)
        epoch_c, epoch_g = 0.0, 0.0
        for step in range(steps_per_epoch):
            batch_idx = idx[step * batch_size : (step + 1) * batch_size]
            if len(batch_idx) < 2:
                continue
            real_x = tf.constant(X_train[batch_idx], dtype=tf.float32)
            labels = tf.constant(y_train[batch_idx].reshape(-1, 1), dtype=tf.int32)
            bs = len(batch_idx)

            c_loss = _train_critic_step(
                critic, generator, real_x, labels, latent_dim, n_critic, gp_weight, opt_c
            )
            g_loss = _train_generator_step(
                critic, generator, labels, latent_dim, bs, opt_g
            )
            epoch_c += float(c_loss)
            epoch_g += float(g_loss)

        history["c_loss"].append(epoch_c / steps_per_epoch)
        history["g_loss"].append(epoch_g / steps_per_epoch)
        if (epoch + 1) % 10 == 0:
            print(f"  GAN epoch {epoch + 1}/{epochs} — c_loss={history['c_loss'][-1]:.4f}, "
                  f"g_loss={history['g_loss'][-1]:.4f}")

    return generator, critic, history


def augment_training_data_with_gan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_synthetic_per_class: int,
    gan_config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training samples per class using trained GAN.

    Trains GAN on X_train/y_train only (no leakage).
    """
    config = gan_config or {}
    latent_dim = config.get("latent_dim", GAN_LATENT_DIM)
    epochs = config.get("epochs", GAN_EPOCHS)
    batch_size = config.get("batch_size", GAN_BATCH_SIZE)

    print("Training WGAN-GP on training data only...")
    generator, _, _ = train_wgan_gp(
        X_train, y_train,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        n_critic=config.get("n_critic", 5),
        gp_weight=config.get("gp_weight", 10.0),
    )

    classes = np.unique(y_train)
    synthetic_X, synthetic_y = [], []
    for cls in classes:
        noise = np.random.randn(num_synthetic_per_class, latent_dim).astype(np.float32)
        labels = np.full((num_synthetic_per_class, 1), cls, dtype=np.int32)
        fake = generator.predict([noise, labels], verbose=0)
        synthetic_X.append(fake)
        synthetic_y.append(np.full(num_synthetic_per_class, cls))

    X_aug = np.concatenate([X_train] + synthetic_X, axis=0)
    y_aug = np.concatenate([y_train] + synthetic_y, axis=0)
    perm = np.random.permutation(len(y_aug))
    return X_aug[perm], y_aug[perm]


def compare_real_vs_synthetic_stats(
    X_real: np.ndarray,
    X_syn: np.ndarray,
) -> dict[str, float]:
    """Basic distribution comparison between real and synthetic EEG."""
    return {
        "real_mean": float(np.mean(X_real)),
        "real_std": float(np.std(X_real)),
        "syn_mean": float(np.mean(X_syn)),
        "syn_std": float(np.std(X_syn)),
        "mean_diff": float(abs(np.mean(X_real) - np.mean(X_syn))),
        "std_diff": float(abs(np.std(X_real) - np.std(X_syn))),
    }
