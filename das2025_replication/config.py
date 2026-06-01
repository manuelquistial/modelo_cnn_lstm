"""
Global configuration for Das et al. (2025) replication experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

# PhysioNet EEG Motor Movement/Imagery Dataset
SFREQ = 160.0
N_CHANNELS_FULL = 64
EXCLUDED_SUBJECTS = [38, 88, 89, 92, 100, 104]
ALL_SUBJECTS = list(range(1, 110))

# Motor imagery runs (article)
MI_RUNS = [4, 6, 8, 10, 12, 14]
BINARY_RUNS = [4, 8, 12]
MULTICLASS_RUNS = [4, 6, 8, 10, 12, 14]

# Filtering (article)
DEFAULT_L_FREQ = 0.5
DEFAULT_H_FREQ = 50.0

# Primary segment length (article: 5 s)
DEFAULT_SEGMENT_LENGTH = 5.0
SEGMENT_LENGTHS_COMPARISON = [1.0, 4.0, 5.0]

# ROI definitions (Das et al. 2025) — unique channels only
ROI_DEFINITIONS: dict[str, list[str]] = {
    "ROI_1": ["FC1", "FC2", "FC3", "FC4", "FC5", "FC6"],
    "ROI_2": ["C5", "C6", "C3", "C4", "C1", "C2"],
    "ROI_3": ["CP1", "CP2", "CP3", "CP4", "CP5", "CP6"],
    "ROI_4": [
        "FC3", "FC4", "C5", "C6", "C3", "C4", "C1", "C2", "CP3", "CP4",
    ],
    "ROI_5": ["FC1", "FC2", "FC3", "FC4", "CP1", "CP2", "CP3", "CP4"],
    "ROI_6": [
        "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
        "C5", "C6", "C3", "C4", "C1", "C2",
        "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
    ],
}

# Class names — Das et al. Table 7 labels (Mode A: 5-class)
PAPER_CLASS_NAMES = ["E", "F", "G", "H", "I"]
PAPER_CLASS_DESCRIPTIONS = {
    "E": "imagined_left_fist",      # runs 4,8,12 T1
    "F": "imagined_both_fists",     # runs 6,10,14 T1
    "G": "imagined_right_fist",     # runs 4,8,12 T2
    "H": "imagined_both_feet",      # runs 6,10,14 T2
    "I": "baseline_rest",           # T0 all runs
}

# Mode B: binary left vs right (runs 4, 8, 12 only)
BINARY_CLASS_NAMES = ["left_hand", "right_hand"]

# Mode A multiclass (alias internal names → paper labels E–I)
MULTICLASS_CLASS_NAMES = PAPER_CLASS_NAMES
MULTICLASS_INTERNAL_NAMES = [
    "left_fist", "both_fists", "right_fist", "both_feet", "rest",
]

# Label maps (PhysioNet T0/T1/T2 × run → class index)
BINARY_LABEL_MAP = {"left_hand": 0, "right_hand": 1}
MULTICLASS_LABEL_MAP = {
    "left_fist": 0,    # E — runs 4,8,12 T1
    "both_fists": 1,   # F — runs 6,10,14 T1
    "right_fist": 2,   # G — runs 4,8,12 T2
    "both_feet": 3,    # H — runs 6,10,14 T2
    "rest": 4,         # I — T0
}
PAPER_LABEL_FROM_INTERNAL = {
    "left_fist": "E",
    "both_fists": "F",
    "right_fist": "G",
    "both_feet": "H",
    "rest": "I",
}

# Reproducibility & splits (thesis-grade: by subject, no trial leakage)
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
TRAIN_SIZE = 0.70
# Legacy 80/20 two-way split when three_way_split=False
LEGACY_TEST_SIZE = 0.2

# Deep learning defaults
DL_EPOCHS = 50
DL_BATCH_SIZE = 64
DL_LEARNING_RATE = 1e-3

# GAN defaults (article)
GAN_EPOCHS = 100
GAN_BATCH_SIZE = 64
GAN_LATENT_DIM = 100

# Output paths
DEFAULT_OUTPUT_DIR = Path("outputs/das2025_replication")
FIGURES_DIR_NAME = "figures"


def get_paper_subjects() -> list[int]:
    """Return valid subject IDs (103 subjects after exclusions)."""
    excluded = set(EXCLUDED_SUBJECTS)
    subjects = [s for s in ALL_SUBJECTS if s not in excluded]
    assert len(subjects) == 103, f"Expected 103 subjects, got {len(subjects)}"
    return subjects


def get_output_paths(output_dir: str | Path | None = None) -> dict[str, Path]:
    """Create and return standard output directory paths."""
    base = Path(output_dir or DEFAULT_OUTPUT_DIR)
    figures = base / FIGURES_DIR_NAME
    base.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return {"base": base, "figures": figures}


def save_experiment_config(config: dict, output_dir: Path) -> Path:
    """Persist experiment configuration as JSON."""
    path = output_dir / "experiment_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
    return path
