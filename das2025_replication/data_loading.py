"""
PhysioNet EEG Motor Movement/Imagery dataset loading, event mapping, and epoching.
"""

from __future__ import annotations

import warnings
from typing import Literal

import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci
from mne.io import read_raw_edf

from .config import (
    BINARY_LABEL_MAP,
    BINARY_RUNS,
    DEFAULT_H_FREQ,
    DEFAULT_L_FREQ,
    MULTICLASS_LABEL_MAP,
    MULTICLASS_RUNS,
    get_paper_subjects,
)

ModeType = Literal["binary", "multiclass"]

EVENT_CODE_T0 = 1
EVENT_CODE_T1 = 2
EVENT_CODE_T2 = 3


def _standardize_channel_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    return raw


def load_physionet_subject(
    subject_id: int,
    runs: list[int],
    verbose: bool | str = False,
) -> mne.io.BaseRaw:
    """Load and concatenate selected PhysioNet runs for one subject."""
    if verbose is False:
        mne.set_log_level("WARNING")

    raw_fnames = eegbci.load_data(subject=subject_id, runs=runs, verbose=verbose)
    raws = []
    for run_id, fname in zip(runs, raw_fnames):
        raw = read_raw_edf(fname, preload=True, verbose=verbose)
        raw = _standardize_channel_names(raw)
        raws.append(raw)

    if len(raws) == 1:
        combined = raws[0]
    else:
        combined, _ = mne.concatenate_raws(raws)
    combined.info["subject_id"] = subject_id
    return combined


def _binary_class_for_run(run_id: int, event_code: int) -> tuple[int, str] | None:
    if run_id not in BINARY_RUNS:
        return None
    if event_code == EVENT_CODE_T1:
        return BINARY_LABEL_MAP["left_hand"], "left_hand"
    if event_code == EVENT_CODE_T2:
        return BINARY_LABEL_MAP["right_hand"], "right_hand"
    return None


def _multiclass_class_for_run(
    run_id: int, event_code: int
) -> tuple[int, str] | None:
    if run_id in (4, 8, 12):
        if event_code == EVENT_CODE_T1:
            return MULTICLASS_LABEL_MAP["left_fist"], "left_fist"
        if event_code == EVENT_CODE_T2:
            return MULTICLASS_LABEL_MAP["right_fist"], "right_fist"
    elif run_id in (6, 10, 14):
        if event_code == EVENT_CODE_T1:
            return MULTICLASS_LABEL_MAP["both_fists"], "both_fists"
        if event_code == EVENT_CODE_T2:
            return MULTICLASS_LABEL_MAP["both_feet"], "both_feet"
    if event_code == EVENT_CODE_T0:
        return MULTICLASS_LABEL_MAP["rest"], "rest"
    return None


def map_events_binary(events: np.ndarray, run_id: int) -> list[dict]:
    """Map MNE events to binary left/right labels."""
    mapped = []
    for trial_id, row in enumerate(events):
        event_code = int(row[2])
        result = _binary_class_for_run(run_id, event_code)
        if result is None:
            continue
        label, class_name = result
        mapped.append({
            "run": run_id,
            "trial_id": trial_id,
            "original_event": event_code,
            "mapped_label": label,
            "class_name": class_name,
            "event_onset": int(row[0]),
            "event_duration": None,
        })
    return mapped


def map_events_multiclass(events: np.ndarray, run_id: int) -> list[dict]:
    """Map MNE events to five-class labels."""
    mapped = []
    for trial_id, row in enumerate(events):
        event_code = int(row[2])
        result = _multiclass_class_for_run(run_id, event_code)
        if result is None:
            continue
        label, class_name = result
        mapped.append({
            "run": run_id,
            "trial_id": trial_id,
            "original_event": event_code,
            "mapped_label": label,
            "class_name": class_name,
            "event_onset": int(row[0]),
            "event_duration": None,
        })
    return mapped


def _label_for_event(mode: ModeType, run_id: int, event_code: int):
    if mode == "binary":
        return _binary_class_for_run(run_id, event_code)
    return _multiclass_class_for_run(run_id, event_code)


def extract_epochs_for_subject(
    subject_id: int,
    runs: list[int],
    mode: ModeType = "binary",
    segment_length: float = 5.0,
    tmin: float = 0.0,
    l_freq: float = DEFAULT_L_FREQ,
    h_freq: float = DEFAULT_H_FREQ,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """
    Load, filter, and epoch EEG for one subject.

    Returns X (n_trials, n_channels, n_samples), y, metadata, channel_names.
    """
    if mode == "binary":
        runs = [r for r in runs if r in BINARY_RUNS] or list(BINARY_RUNS)
    else:
        runs = [r for r in runs if r in MULTICLASS_RUNS] or list(MULTICLASS_RUNS)

    all_X: list[np.ndarray] = []
    all_y: list[int] = []
    meta_rows: list[dict] = []
    ch_names: list[str] = []

    for run_id in runs:
        raw = load_physionet_subject(subject_id, [run_id], verbose=verbose)
        raw.filter(l_freq, h_freq, fir_design="firwin", verbose=verbose)
        sfreq = raw.info["sfreq"]
        tmax = tmin + segment_length - (1.0 / sfreq)

        events, _ = mne.events_from_annotations(raw, verbose=False)

        # Build event_id for epoching: only events with valid labels
        event_id: dict[str, int] = {}
        for row in events:
            code = int(row[2])
            info = _label_for_event(mode, run_id, code)
            if info is not None:
                _, cname = info
                event_id[cname] = code

        if not event_id:
            continue

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=verbose,
            event_repeated="drop",
        )
        if len(epochs) == 0:
            continue

        if not ch_names:
            ch_names = epochs.ch_names

        data = epochs.get_data()
        ep_events = epochs.events

        for i in range(len(epochs)):
            code = int(ep_events[i, 2])
            label_info = _label_for_event(mode, run_id, code)
            if label_info is None:
                continue
            label, class_name = label_info
            all_X.append(data[i])
            all_y.append(label)
            meta_rows.append({
                "subject": subject_id,
                "run": run_id,
                "trial_id": i,
                "original_event": code,
                "class_name": class_name,
                "label": label,
                "epoch_start_time": ep_events[i, 0] / sfreq,
                "segment_length": segment_length,
                "sfreq": sfreq,
                "n_samples": data.shape[-1],
            })

    if not all_X:
        return np.zeros((0, 0, 0)), np.array([], dtype=int), pd.DataFrame(), []

    X = np.stack(all_X, axis=0)
    y = np.array(all_y, dtype=int)
    metadata = pd.DataFrame(meta_rows)

    assert len(X) == len(y) == len(metadata)
    assert X.ndim == 3
    required = [
        "subject", "run", "trial_id", "original_event", "class_name",
        "label", "epoch_start_time", "segment_length", "sfreq", "n_samples",
    ]
    for col in required:
        assert col in metadata.columns, f"Missing column: {col}"

    return X, y, metadata, ch_names


def build_dataset(
    subjects: list[int] | None = None,
    runs: list[int] | None = None,
    mode: ModeType = "binary",
    segment_length: float = 5.0,
    roi_name: str = "ROI_6",
    l_freq: float = DEFAULT_L_FREQ,
    h_freq: float = DEFAULT_H_FREQ,
    paper_input: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Build full dataset across subjects with ROI channel selection."""
    from .rois import select_roi

    if subjects is None:
        subjects = get_paper_subjects()
    if runs is None:
        runs = BINARY_RUNS if mode == "binary" else MULTICLASS_RUNS

    X_list, y_list, meta_list = [], [], []
    channel_names: list[str] = []

    for i, subj in enumerate(subjects):
        if verbose and i % 10 == 0:
            print(f"  Loading subject {subj} ({i + 1}/{len(subjects)})...")
        try:
            X_s, y_s, meta_s, ch_names = extract_epochs_for_subject(
                subj, runs, mode=mode, segment_length=segment_length,
                l_freq=l_freq, h_freq=h_freq, verbose=False,
            )
        except Exception as exc:
            warnings.warn(f"Skipping subject {subj}: {exc}")
            continue
        if len(X_s) == 0:
            continue
        if not channel_names:
            channel_names = ch_names
        X_list.append(X_s)
        y_list.append(y_s)
        meta_list.append(meta_s)

    if not X_list:
        raise RuntimeError("No data loaded for any subject.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    metadata = pd.concat(meta_list, ignore_index=True)
    X, roi_channel_names = select_roi(X, channel_names, roi_name)

    if paper_input:
        from .paper_input import to_paper_input_shape

        X, pair = to_paper_input_shape(X, roi_channel_names, roi_name)
        roi_channel_names = list(pair)
        if verbose:
            print(
                f"  Paper input: contralateral pair {pair}, shape {X.shape} "
                f"(trials, samples, 2)"
            )

    assert len(X) == len(y) == len(metadata)
    return X, y, metadata, roi_channel_names


def ensure_channel_first(X: np.ndarray) -> np.ndarray:
    """Ensure shape (n_trials, n_channels, n_samples) for ML/CSP/Riemannian."""
    if X.size == 0:
        return X
    if X.ndim == 3 and X.shape[-1] <= 8 and X.shape[1] > X.shape[-1]:
        return np.transpose(X, (0, 2, 1))
    return X


def prepare_deep_learning_input(X: np.ndarray) -> np.ndarray:
    """
    Prepare tensor for Conv1D models.

    (n_trials, n_channels, n_samples) -> (n_trials, n_samples, n_channels)
    If already paper format (n_trials, n_samples, 2), returns unchanged.
    """
    if X.size == 0:
        return X
    if X.ndim == 3 and X.shape[-1] <= 4 and X.shape[1] > X.shape[-1]:
        # Already (trials, samples, channels) e.g. paper 640 x 2
        return X
    return np.transpose(X, (0, 2, 1))


def print_dataset_summary(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    class_names: list[str],
) -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"X shape:          {X.shape}")
    print(f"y shape:          {y.shape}")
    print(f"N trials:         {len(y)}")
    print(f"N subjects:       {metadata['subject'].nunique()}")
    print(f"N channels:       {X.shape[1] if X.ndim == 3 else 'N/A'}")
    print(f"N samples/trial:  {X.shape[2] if X.ndim == 3 else 'N/A'}")
    print("\nClass distribution:")
    for u, c in zip(*np.unique(y, return_counts=True)):
        name = class_names[int(u)] if int(u) < len(class_names) else str(u)
        print(f"  {name} (label={u}): {c}")
    print("\nTrials per subject (first 10):")
    print(metadata.groupby("subject").size().head(10).to_string())
    print("\nTrials per run:")
    print(metadata.groupby("run").size().to_string())
    print("=" * 60 + "\n")
