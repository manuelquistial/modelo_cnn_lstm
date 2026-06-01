"""
Pre-processing aligned with Das et al. (2025): ICA, CSP spatial filtering, paper DL input.

Paper pipeline (Methodology): normalization, band-pass 0.5–50 Hz, CSP, ICA artifact
removal, then 640×2 contralateral matrix (5 s epochs center-cropped to 640 samples).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .data_loading import ensure_channel_first
from .paper_input import PAPER_MATRIX_SAMPLES, to_paper_input_shape


def _apply_csp_filters(X: np.ndarray, filters: np.ndarray) -> np.ndarray:
    return np.einsum("ci,tij->tcj", filters, ensure_channel_first(X))


def fit_apply_csp_spatial_filter(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, Any]:
    """CSP spatial filtering (Eq. 3) — fit on train only."""
    from mne.decoding import CSP

    X_train = ensure_channel_first(X_train)
    X_test = ensure_channel_first(X_test)
    n_ch = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    max_comp = min(n_ch - 1, max(1, n_classes - 1))
    n_comp = min(n_components or 4, max_comp)
    if n_comp < 1:
        return X_train, X_test, n_ch, None

    csp = CSP(n_components=n_comp, reg="oas", log=False, norm_trace=False)
    csp.fit(X_train, y_train)
    X_tr = _apply_csp_filters(X_train, csp.filters_)
    X_te = _apply_csp_filters(X_test, csp.filters_)
    return X_tr, X_te, n_comp, csp


def apply_csp_spatial_filter(X: np.ndarray, csp: Any) -> np.ndarray:
    """Apply fitted CSP to validation/test (no refit)."""
    if csp is None:
        return ensure_channel_first(X)
    return _apply_csp_filters(X, csp.filters_)


def _crop_to_paper_samples(X: np.ndarray) -> np.ndarray:
    """Center-crop or pad to 640 samples (paper matrix width)."""
    if X.ndim == 3 and X.shape[-1] <= 8:
        target = PAPER_MATRIX_SAMPLES
        n = X.shape[1]
        if n > target:
            start = (n - target) // 2
            return X[:, start : start + target, :]
        if n < target:
            return np.pad(X, ((0, 0), (0, target - n), (0, 0)), mode="edge")
        return X
    target = PAPER_MATRIX_SAMPLES
    n = X.shape[2]
    if n > target:
        start = (n - target) // 2
        return X[:, :, start : start + target]
    if n < target:
        return np.pad(X, ((0, 0), (0, 0), (0, target - n)), mode="edge")
    return X


def _csp_to_paper_matrix(X_csp: np.ndarray) -> np.ndarray:
    """Top-2 CSP components → (trials, 640, 2)."""
    X_pair = np.transpose(X_csp[:, :2, :], (0, 2, 1))
    return _crop_to_paper_samples(X_pair)


def prepare_paper_dl_tensors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    channel_names: list[str],
    roi_name: str,
    *,
    paper_input: bool,
    paper_preprocess: bool,
    X_val: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    """Build train/val/test DL tensors (CSP fit on train only)."""
    X_tr = ensure_channel_first(X_train)
    X_te = ensure_channel_first(X_test)
    X_va = ensure_channel_first(X_val) if X_val is not None else None
    out_names = list(channel_names)

    if paper_preprocess:
        X_tr, X_te, n_comp, csp = fit_apply_csp_spatial_filter(X_tr, y_train, X_te)
        if X_va is not None:
            X_va = apply_csp_spatial_filter(X_va, csp)
        out_names = [f"CSP{i + 1}" for i in range(n_comp)]
        if paper_input:
            if n_comp < 2:
                warnings.warn("CSP n_components < 2; using contralateral pair.")
                X_tr, _ = to_paper_input_shape(X_tr, channel_names, roi_name)
                X_te, _ = to_paper_input_shape(X_te, channel_names, roi_name)
                if X_va is not None:
                    X_va, _ = to_paper_input_shape(X_va, channel_names, roi_name)
            else:
                X_tr = _csp_to_paper_matrix(X_tr)
                X_te = _csp_to_paper_matrix(X_te)
                if X_va is not None:
                    X_va = _csp_to_paper_matrix(X_va)
                out_names = ["CSP1", "CSP2"]
            return X_tr, X_te, X_va, out_names

    if paper_input:
        X_tr, pair = to_paper_input_shape(X_tr, channel_names, roi_name)
        X_te, _ = to_paper_input_shape(X_te, channel_names, roi_name)
        if X_va is not None:
            X_va, _ = to_paper_input_shape(X_va, channel_names, roi_name)
        return X_tr, X_te, X_va, list(pair)

    return X_tr, X_te, X_va, out_names


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
    X_norm = X_train if X_train.shape[1] > X_train.shape[-1] else X_train

    n_per_class = max(50, len(y_train) // (len(np.unique(y_train)) * 2))
    return augment_training_data_with_gan(
        X_norm, y_train, num_synthetic_per_class=n_per_class, gan_config=gan_config
    )
