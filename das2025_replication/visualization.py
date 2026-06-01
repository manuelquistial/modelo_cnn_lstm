"""
Plotting utilities for Das et al. 2025 replication experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix
from sklearn.preprocessing import label_binarize


def _save_or_show(save_path: str | Path | None) -> None:
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title)
    _save_or_show(save_path)


def plot_roc_curve_binary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    save_path: str | Path | None = None,
) -> None:
    if y_proba.ndim == 2 and y_proba.shape[1] > 1:
        scores = y_proba[:, 1]
    else:
        scores = y_proba
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_true, scores, ax=ax)
    ax.set_title(title)
    _save_or_show(save_path)


def plot_training_history(
    history: dict[str, list],
    title: str = "Training History",
    save_path: str | Path | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "loss" in history:
        axes[0].plot(history["loss"], label="train")
        if "val_loss" in history:
            axes[0].plot(history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()
    if "accuracy" in history:
        axes[1].plot(history["accuracy"], label="train")
        if "val_accuracy" in history:
            axes[1].plot(history["val_accuracy"], label="val")
        axes[1].set_title("Accuracy")
        axes[1].legend()
    fig.suptitle(title)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "accuracy",
    title: str = "Model Comparison",
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    models = results_df["model"].values if "model" in results_df.columns else results_df.index
    values = results_df[metric].values
    ax.bar(range(len(models)), values)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_roi_comparison(
    roi_results_df: pd.DataFrame,
    metric: str = "accuracy",
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if "roi" in roi_results_df.columns and "model" in roi_results_df.columns:
        pivot = roi_results_df.pivot_table(
            index="roi", columns="model", values=metric, aggfunc="mean"
        )
        pivot.plot(kind="bar", ax=ax)
    else:
        ax.bar(roi_results_df["roi"], roi_results_df[metric])
    ax.set_ylabel(metric)
    ax.set_title("ROI Comparison")
    ax.legend(loc="best")
    plt.tight_layout()
    _save_or_show(save_path)


def plot_errors_by_subject(
    errors_by_subject_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    df = errors_by_subject_df.sort_values("accuracy")
    ax.bar(df["subject"].astype(str), df["accuracy"])
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Subject")
    plt.xticks(rotation=90)
    plt.tight_layout()
    _save_or_show(save_path)


def run_pca_visualization(
    features: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    title: str = "PCA",
    save_path: str | Path | None = None,
    n_components: int = 2,
) -> np.ndarray:
    """PCA for visualization only."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(features)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in np.unique(y):
        mask = y == label
        name = class_names[int(label)] if int(label) < len(class_names) else str(label)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, alpha=0.6, s=15)
    ax.set_title(title)
    ax.legend()
    _save_or_show(save_path)
    return X_pca


def plot_tsne_raw_features(
    X_features: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    title: str = "t-SNE (raw features)",
    save_path: str | Path | None = None,
    perplexity: float = 30.0,
) -> np.ndarray | None:
    """t-SNE on feature vectors (visualization only)."""
    n = len(y)
    if n < 4:
        print(f"  t-SNE skipped: need >= 4 samples, got {n}.", flush=True)
        return None
    if len(y) != X_features.shape[0]:
        n = min(len(y), X_features.shape[0])
        X_features = X_features[:n]
        y = y[:n]

    perp = min(perplexity, max(5, (n - 1) // 3))
    X_in = X_features
    n_samples, n_features = X_features.shape
    max_pca = min(50, n_features, max(2, n_samples - 1))
    if n_features > max_pca and max_pca >= 2:
        print(
            f"  t-SNE: PCA {max_pca}D pre-reduction ({n_features} features, {n_samples} samples)...",
            flush=True,
        )
        X_in = PCA(n_components=max_pca, random_state=42).fit_transform(X_features)
    print(f"  t-SNE: fitting {n} samples (perplexity={perp:.0f})...", flush=True)
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perp,
        max_iter=500,
        init="pca",
        learning_rate="auto",
    )
    X_emb = tsne.fit_transform(X_in)
    print("  t-SNE: done.", flush=True)
    _plot_tsne_scatter(X_emb, y, class_names, title, save_path)
    return X_emb


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    title: str = "t-SNE (embeddings)",
    save_path: str | Path | None = None,
    perplexity: float = 30.0,
) -> np.ndarray:
    """t-SNE on deep model embeddings."""
    return plot_tsne_raw_features(
        embeddings, y, class_names, title=title, save_path=save_path, perplexity=perplexity
    )


def _plot_tsne_scatter(
    X_emb: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: str | Path | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in np.unique(y):
        mask = y == label
        name = class_names[int(label)] if int(label) < len(class_names) else str(label)
        ax.scatter(X_emb[mask, 0], X_emb[mask, 1], label=name, alpha=0.6, s=15)
    ax.set_title(title)
    ax.legend()
    _save_or_show(save_path)


def plot_real_vs_synthetic_eeg(
    real_samples: np.ndarray,
    synthetic_samples: np.ndarray,
    channel_idx: int = 0,
    save_path: str | Path | None = None,
) -> None:
    """Plot example real vs synthetic EEG traces."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(real_samples[0, :, channel_idx], label="real")
    axes[0].set_title("Real EEG sample")
    axes[1].plot(synthetic_samples[0, :, channel_idx], label="synthetic", color="orange")
    axes[1].set_title("Synthetic EEG sample")
    axes[1].set_xlabel("Time samples")
    plt.tight_layout()
    _save_or_show(save_path)
