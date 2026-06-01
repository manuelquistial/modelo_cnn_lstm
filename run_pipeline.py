#!/usr/bin/env python3
"""
Entry point for Das et al. (2025) EEG motor imagery replication.

Usage:
  python run_pipeline.py --quick
  python run_pipeline.py --max-subjects 20 --epochs 30
  python run_pipeline.py --gan
"""

import argparse
import sys
from pathlib import Path

from das2025_replication.config import DEFAULT_OUTPUT_DIR, DL_EPOCHS
from das2025_replication.run_experiments import run_complete_das2025_replication

_REPO = Path(__file__).resolve().parent
_VENV_PY = _REPO / ".venv" / "bin" / "python"


def _warn_if_not_venv() -> None:
    in_venv = hasattr(sys, "real_prefix") or sys.prefix != sys.base_prefix
    if not in_venv and _VENV_PY.is_file():
        print(
            "AVISO: no estás dentro de .venv. Usa:\n"
            f"  source {_REPO}/.venv/bin/activate\n"
            f"  o: {_VENV_PY} {Path(__file__).name} ...",
            file=sys.stderr,
        )

if __name__ == "__main__":
    _warn_if_not_venv()
    parser = argparse.ArgumentParser(
        description="Das et al. (2025) EEG MI classification replication"
    )
    parser.add_argument("--quick", action="store_true", help="5 subjects, skip ROI/segment")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=DL_EPOCHS)
    parser.add_argument("--gan", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    run_complete_das2025_replication(
        max_subjects=5 if args.quick else args.max_subjects,
        dl_epochs=args.epochs,
        run_gan=args.gan,
        do_roi_experiments=not args.quick,
        do_segment_experiment=not args.quick,
        output_dir=args.output_dir,
    )
