"""
Data splitting, normalization, ICA, and CSP spatial filtering.
"""

from __future__ import annotations

import warnings
from typing import Literal

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    LEGACY_TEST_SIZE,
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)

SplitStrategy = Literal["trialwise", "subjectwise", "subjectwise_3way"]


def make_subjectwise_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    train_ratio: float = TRAIN_SIZE,
    val_ratio: float = VAL_SIZE,
    test_ratio: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, ...]:
    """
    Subject-wise 70/15/15 split (train / validation / test).

    No subject appears in more than one partition — recommended for thesis evaluation.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    groups = metadata["subject"].values
    idx = np.arange(len(y))

    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=random_state
    )
    trainval_idx, test_idx = next(gss_test.split(X, y, groups=groups))

    groups_tv = metadata.iloc[trainval_idx]["subject"].values
    val_share = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_share, random_state=random_state + 1
    )
    tr_rel, val_rel = next(
        gss_val.split(X[trainval_idx], y[trainval_idx], groups=groups_tv)
    )
    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[val_rel]

    _assert_no_subject_leakage(metadata, train_idx, val_idx, test_idx)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    meta_train = metadata.iloc[train_idx].reset_index(drop=True)
    meta_val = metadata.iloc[val_idx].reset_index(drop=True)
    meta_test = metadata.iloc[test_idx].reset_index(drop=True)

    print("\n--- Split summary (subject-wise 70/15/15) ---")
    print(
        f"Train: {len(y_train)} trials, {meta_train['subject'].nunique()} subjects"
    )
    print(f"Val:   {len(y_val)} trials, {meta_val['subject'].nunique()} subjects")
    print(f"Test:  {len(y_test)} trials, {meta_test['subject'].nunique()} subjects")
    print("Train classes:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Test classes:", dict(zip(*np.unique(y_test, return_counts=True))))
    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        meta_train, meta_val, meta_test,
    )


def _assert_no_subject_leakage(
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    tr = set(metadata.iloc[train_idx]["subject"].unique())
    va = set(metadata.iloc[val_idx]["subject"].unique())
    te = set(metadata.iloc[test_idx]["subject"].unique())
    assert not (tr & va), f"Train/val subject leak: {tr & va}"
    assert not (tr & te), f"Train/test subject leak: {tr & te}"
    assert not (va & te), f"Val/test subject leak: {va & te}"


def make_trialwise_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    test_size: float = LEGACY_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, ...]:
    """Stratified random split by trial — may leak subjects; not for primary thesis metrics."""
    warnings.warn(
        "Trial-wise split may inflate accuracy (same subject in train and test). "
        "Use subjectwise_3way for thesis evaluation.",
        UserWarning,
        stacklevel=2,
    )
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    result = _split_by_indices(X, y, metadata, train_idx, test_idx)
    _print_split_summary(*result, split_name="trial-wise")
    return result


def make_subjectwise_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    test_size: float = LEGACY_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, ...]:
    """Split by subject — no subject in both train and test."""
    subjects = metadata["subject"].unique()
    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    groups = metadata["subject"].values
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    train_subjects = set(metadata.iloc[train_idx]["subject"].unique())
    test_subjects = set(metadata.iloc[test_idx]["subject"].unique())
    overlap = train_subjects & test_subjects
    assert len(overlap) == 0, f"Subject leakage detected: {overlap}"

    result = _split_by_indices(X, y, metadata, train_idx, test_idx)
    _print_split_summary(*result, split_name="subject-wise")
    return result


def make_group_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, ...]]:
    """Group K-fold splits by subject."""
    groups = metadata["subject"].values
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        splits.append(_split_by_indices(X, y, metadata, train_idx, test_idx))
    return splits


def _split_by_indices(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    meta_train = metadata.iloc[train_idx].reset_index(drop=True)
    meta_test = metadata.iloc[test_idx].reset_index(drop=True)

    assert len(X_train) == len(y_train) == len(meta_train)
    assert len(X_test) == len(y_test) == len(meta_test)
    return X_train, X_test, y_train, y_test, meta_train, meta_test


def _print_split_summary(
    X_train, X_test, y_train, y_test, meta_train, meta_test, split_name: str = ""
) -> None:
    """Print train/test split statistics."""
    print(f"\n--- Split summary ({split_name}) ---")
    print(f"Train trials: {len(y_train)}, Test trials: {len(y_test)}")
    print(f"Train subjects: {meta_train['subject'].nunique()}, "
          f"Test subjects: {meta_test['subject'].nunique()}")
    train_subs = set(meta_train["subject"].unique())
    test_subs = set(meta_test["subject"].unique())
    overlap = train_subs & test_subs
    print(f"Subject overlap: {overlap if overlap else 'None (OK)'}")
    print("Train class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Test class distribution:", dict(zip(*np.unique(y_test, return_counts=True))))


def make_validation_split_from_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
    subjectwise: bool = True,
) -> tuple[np.ndarray, ...]:
    """Create validation set from training data."""
    if subjectwise and "subject" in meta_train.columns:
        groups = meta_train["subject"].values
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        tr_idx, val_idx = next(gss.split(X_train, y_train, groups=groups))
    else:
        idx = np.arange(len(y_train))
        tr_idx, val_idx = train_test_split(
            idx, test_size=val_size, stratify=y_train, random_state=random_state
        )
    return _split_by_indices(X_train, y_train, meta_train, tr_idx, val_idx)


def fit_channelwise_standardizer(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit channel-wise z-score on training data.

    X_train shape: (n_trials, n_samples, n_channels)
    Returns mean (1, 1, n_channels), std (1, 1, n_channels)
    """
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_channelwise_standardizer(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply channel-wise z-score normalization."""
    return (X - mean) / std


def fit_sklearn_scaler(X_train_features: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on training features only."""
    scaler = StandardScaler()
    scaler.fit(X_train_features)
    return scaler


def apply_sklearn_scaler(
    scaler: StandardScaler, X_features: np.ndarray
) -> np.ndarray:
    """Transform features with fitted scaler."""
    return scaler.transform(X_features)


def apply_ica_artifact_removal(
    raw: mne.io.BaseRaw,
    n_components: float | int | None = 0.99,
    random_state: int = RANDOM_STATE,
    auto_reject: bool = True,
) -> mne.io.BaseRaw:
    """
    ICA artifact removal on Raw data (Das et al. 2025, Eq. 4).

    When ``auto_reject=True``, excludes up to two components with highest
    kurtosis (muscle/artifact proxy) when EOG channels are unavailable.
    """
    raw_ica = raw.copy()
    n_comp = n_components
    if n_comp is None:
        n_comp = min(int(0.99 * len(mne.pick_types(raw_ica.info, eeg=True))), 20)
    ica = mne.preprocessing.ICA(
        n_components=n_comp,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw_ica)
    if auto_reject:
        excluded: list[int] = []
        try:
            eog_indices, _ = ica.find_bads_eog(raw_ica, verbose=False)
            excluded.extend(eog_indices)
        except Exception:
            pass
        if not excluded:
            try:
                muscle, _ = ica.find_bads_muscle(raw_ica, verbose=False)
                excluded.extend(muscle[:2])
            except Exception:
                pass
        if not excluded and hasattr(ica, "get_sources"):
            try:
                sources = ica.get_sources(raw_ica).get_data()
                kurt = np.mean(
                    (sources - sources.mean(axis=1, keepdims=True)) ** 4,
                    axis=1,
                ) / (
                    np.var(sources, axis=1) ** 2 + 1e-12
                )
                excluded = np.argsort(kurt)[-2:].tolist()
            except Exception:
                pass
        ica.exclude = list(dict.fromkeys(excluded))[:2]
    ica.apply(raw_ica)
    return raw_ica


def extract_csp_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CSP spatial filtering — fit only on training data.

    X shape: (n_trials, n_channels, n_samples)
    """
    from mne.decoding import CSP

    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    X_train_csp = csp.fit_transform(X_train, y_train)
    X_test_csp = csp.transform(X_test)
    return X_train_csp, X_test_csp
