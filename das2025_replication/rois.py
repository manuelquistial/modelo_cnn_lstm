"""
Region-of-interest (ROI) channel selection for EEG motor imagery.
"""

from __future__ import annotations

import numpy as np

from .config import ROI_DEFINITIONS


def get_roi_channel_names(roi_name: str) -> list[str]:
    """Return unique channel names for a ROI (no duplicates)."""
    if roi_name not in ROI_DEFINITIONS:
        raise ValueError(
            f"Unknown ROI '{roi_name}'. Available: {list(ROI_DEFINITIONS.keys())}"
        )
    channels = ROI_DEFINITIONS[roi_name]
    unique = list(dict.fromkeys(channels))  # preserve order, remove duplicates
    if len(unique) != len(channels):
        print(
            f"Note: ROI '{roi_name}' had duplicate channel entries in config; "
            f"using {len(unique)} unique channels."
        )
    return unique


def select_roi(
    X: np.ndarray,
    channel_names: list[str],
    roi_name: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Select ROI channels from epoch data.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_samples)
    channel_names : list of str
    roi_name : str

    Returns
    -------
    X_roi : ndarray, shape (n_trials, n_roi_channels, n_samples)
    selected_channel_names : list of str
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3D, got shape {X.shape}")

    roi_channels = get_roi_channel_names(roi_name)
    name_to_idx = {cn.upper(): i for i, cn in enumerate(channel_names)}

    indices: list[int] = []
    selected: list[str] = []
    for ch in roi_channels:
        key = ch.upper()
        if key not in name_to_idx:
            raise AssertionError(
                f"ROI channel '{ch}' not found in dataset channels. "
                f"Available (sample): {channel_names[:15]}..."
            )
        idx = name_to_idx[key]
        indices.append(idx)
        selected.append(channel_names[idx])

    assert len(indices) == len(set(indices)), "Duplicate channel indices in ROI"
    assert len(selected) == len(roi_channels), (
        f"Expected {len(roi_channels)} ROI channels, selected {len(selected)}"
    )

    X_roi = X[:, indices, :]
    assert X_roi.shape[1] == len(roi_channels), (
        f"Output channel count {X_roi.shape[1]} != expected {len(roi_channels)}"
    )
    return X_roi, selected


def list_all_rois() -> list[str]:
    """Return all ROI names."""
    return list(ROI_DEFINITIONS.keys())
