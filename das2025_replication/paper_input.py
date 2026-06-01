"""
Paper-exact EEG input formatting (Das et al. 2025).

The article states each segment is a matrix of size 640 x 2, where the two
columns are paired contralateral channels (Table 3 / methodology text).
640 samples = 4 seconds at 160 Hz (paper also mentions 5 s epochs — internal inconsistency).
"""

from __future__ import annotations

import numpy as np

from .config import ROI_DEFINITIONS, SFREQ

# Primary contralateral pair per ROI (sensorimotor pairs within each ROI's channels)
ROI_PRIMARY_CONTRALATERAL_PAIR: dict[str, tuple[str, str]] = {
    "ROI_1": ("FC3", "FC4"),
    "ROI_2": ("C3", "C4"),
    "ROI_3": ("CP3", "CP4"),
    "ROI_4": ("C3", "C4"),
    "ROI_5": ("FC3", "FC4"),
    "ROI_6": ("C3", "C4"),
}

# Table 6: training epochs per ROI used in the paper's hybrid model
ROI_TRAINING_EPOCHS_PAPER: dict[str, int] = {
    "ROI_1": 35,
    "ROI_2": 42,
    "ROI_3": 47,
    "ROI_4": 41,
    "ROI_5": 50,
    "ROI_6": 29,
}

# Paper class labels (Table 7): E,F,G,H,I
PAPER_MULTICLASS_NAMES = ["E", "F", "G", "H", "I"]
PAPER_LABEL_TO_CLASS = {
    0: "E",  # left fist (runs 4,8,12 T1)
    1: "F",  # both fists (runs 6,10,14 T1) — paper maps T1 both fists to F
    2: "G",  # right fist
    3: "H",  # both feet
    4: "I",  # rest T0
}

# Paper segment sizes
PAPER_MATRIX_SAMPLES = 640  # 4 s @ 160 Hz
PAPER_MATRIX_CHANNELS = 2


def get_paper_segment_length(use_640_samples: bool = True) -> float:
    """Segment length matching 640-sample matrix (4 s) or 5 s epoch text."""
    return PAPER_MATRIX_SAMPLES / SFREQ if use_640_samples else 5.0


def _channel_index(channel_names: list[str], name: str) -> int:
    lookup = {cn.upper(): i for i, cn in enumerate(channel_names)}
    key = name.upper()
    if key not in lookup:
        raise ValueError(f"Channel {name} not in {channel_names}")
    return lookup[key]


def extract_contralateral_pair(
    X: np.ndarray,
    channel_names: list[str],
    pair: tuple[str, str],
) -> np.ndarray:
    """
  Extract contralateral pair -> (n_trials, n_samples, 2).

    Column 0 = first channel of pair, column 1 = second (paper: 640 x 2).
    """
    i0 = _channel_index(channel_names, pair[0])
    i1 = _channel_index(channel_names, pair[1])
    # (trials, samples, 2)
    return np.stack([X[:, i0, :], X[:, i1, :]], axis=-1)


def to_paper_input_shape(
    X: np.ndarray,
    channel_names: list[str],
    roi_name: str,
) -> tuple[np.ndarray, tuple[str, str]]:
    """
    Convert (n_trials, n_channels, n_samples) to paper format (n_trials, n_samples, 2).

    If n_samples != 640, crops or pads symmetrically to 640 (paper matrix size).
    """
    pair = ROI_PRIMARY_CONTRALATERAL_PAIR.get(roi_name)
    if pair is None:
        raise ValueError(f"No contralateral pair defined for {roi_name}")
    X_pair = extract_contralateral_pair(X, channel_names, pair)

    n_trials, n_samples, n_ch = X_pair.shape
    assert n_ch == 2, f"Expected 2 columns, got {n_ch}"

    target = PAPER_MATRIX_SAMPLES
    if n_samples > target:
        start = (n_samples - target) // 2
        X_pair = X_pair[:, start : start + target, :]
    elif n_samples < target:
        pad = target - n_samples
        X_pair = np.pad(X_pair, ((0, 0), (0, pad), (0, 0)), mode="edge")

    return X_pair, pair


def all_contralateral_pairs_in_roi(roi_name: str) -> list[tuple[str, str]]:
    """List standard L-R pairs whose both channels lie in the ROI."""
    channels = set(ROI_DEFINITIONS[roi_name])
    standard_pairs = [
        ("FC1", "FC2"), ("FC3", "FC4"), ("FC5", "FC6"),
        ("C1", "C2"), ("C3", "C4"), ("C5", "C6"),
        ("CP1", "CP2"), ("CP3", "CP4"), ("CP5", "CP6"),
    ]
    return [(a, b) for a, b in standard_pairs if a in channels and b in channels]
