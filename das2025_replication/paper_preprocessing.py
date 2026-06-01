"""
Pre-processing aligned with Das et al. (2025): ICA, CSP spatial filtering, paper DL input.

Paper pipeline (Methodology): normalization, band-pass 0.5–50 Hz, CSP, ICA artifact
removal, then 640×2 contralateral matrix (5 s epochs center-cropped to 640 samples).
"""

from __future__ import annotations

import warnings

import numpy as np

from .config import SFREQ
from .data_loading import ensure_channel_first
from .paper_input import PAPER_MATRIX_SAMPLES, to_paper_input_shape


def fit_apply_csp_spatial_filter(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    CSP spatial filtering (Eq. 3) — fit on train only.

    Returns filtered time series (trials, n_components, samples).
    """
    from mne.decoding import CSP

    X_train = ensure_channel_first(X_train)
    X_test = ensure_channel_first(X_test)
    n_ch = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    max_comp = min(n_ch - 1, max(1, n_classes - 1))
    n_comp = min(n_components or 4, max_comp)
    if n_comp < 1:
        return X_train, X_test, n_ch

    csp = CSP(n_components=n_comp, reg="oas", log=False, norm_trace=False)
    csp.fit(X_train, y_train)
    filters = csp.filters_
    X_tr = np.einsum("ci,tij->tcj", filters, X_train)
    X_te = np.einsum("ci,tij->tcj", filters, X_test)
    return X_tr, X_te, n_comp


def _crop_to_paper_samples(X: np.ndarray) -> np.ndarray:
    """Center-crop or pad to 640 samples (paper matrix width)."""
    n_samples = X.shape[1] if X.ndim == 3 and X.shape[-1] <= 8 else X.shape[2]
    if X.ndim == 3 and X.shape[-1] <= 8:
        # (trials, samples, channels)
        target = PAPER_MATRIX_SAMPLES
        n = X.shape[1]
        if n > target:
            start = (n - target) // 2
            return X[:, start : start + target, :]
        if n < target:
            pad = target - n
            return np.pad(X, ((0, 0), (0, pad), (0, 0)), mode="edge")
        return X
    # (trials, channels, samples)
    target = PAPER_MATRIX_SAMPLES
    n = X.shape[2]
    if n > target:
        start = (n - target) // 2
        return X[:, :, start : start + target]
    if n < target:
        pad = target - n
        return np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="edge")
    return X


def prepare_paper_dl_tensors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    channel_names: list[str],
    roi_name: str,
    *,
    paper_input: bool,
    paper_preprocess: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build train/test tensors for deep models following the paper workflow.

    With ``paper_preprocess``: CSP on ROI channels, then top-2 CSP filters as the
    640×2 matrix (spatial filtering before 2-column input).

    With ``paper_input`` only: contralateral pair from Table 3 (e.g. C3/C4).
    """
    X_tr = ensure_channel_first(X_train)
    X_te = ensure_channel_first(X_test)
    out_names = list(channel_names)

    if paper_preprocess:
        X_tr, X_te, n_comp = fit_apply_csp_spatial_filter(X_tr, y_train, X_te)
        out_names = [f"CSP{i + 1}" for i in range(n_comp)]
        if paper_input:
            if n_comp < 2:
                warnings.warn("CSP n_components < 2; falling back to contralateral pair.")
                X_tr, _ = to_paper_input_shape(X_tr, channel_names, roi_name)
                X_te, _ = to_paper_input_shape(X_te, channel_names, roi_name)
            else:
                X_tr = np.transpose(X_tr[:, :2, :], (0, 2, 1))
                X_te = np.transpose(X_te[:, :2, :], (0, 2, 1))
                X_tr = _crop_to_paper_samples(X_tr)
                X_te = _crop_to_paper_samples(X_te)
                out_names = ["CSP1", "CSP2"]
            return X_tr, X_te, out_names

    if paper_input:
        X_tr, pair = to_paper_input_shape(X_tr, channel_names, roi_name)
        X_te, _ = to_paper_input_shape(X_te, channel_names, roi_name)
        out_names = list(pair)
        return X_tr, X_te, out_names

    return X_tr, X_te, out_names


def augment_train_with_gan_if_enabled(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_gan: bool,
    gan_config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """GAN augmentation on normalized DL tensors ``(trials, samples, channels)``."""
    if not use_gan:
        return X_train, y_train

    from .gan import augment_training_data_with_gan

    if X_train.ndim != 3:
        return X_train, y_train
    # Already DL layout when time axis is largest
    if X_train.shape[1] <= X_train.shape[-1]:
        from .data_loading import prepare_deep_learning_input
        from .preprocessing import (
            apply_channelwise_standardizer,
            fit_channelwise_standardizer,
        )

        X_dl = prepare_deep_learning_input(X_train)
        mean, std = fit_channelwise_standardizer(X_dl)
        X_norm = apply_channelwise_standardizer(X_dl, mean, std)
    else:
        X_norm = X_train

    n_per_class = max(50, len(y_train) // (len(np.unique(y_train)) * 2))
    X_aug, y_aug = augment_training_data_with_gan(
        X_norm, y_train, num_synthetic_per_class=n_per_class, gan_config=gan_config
    )
    return X_aug, y_aug
