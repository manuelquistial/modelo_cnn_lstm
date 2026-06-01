#!/usr/bin/env python3
"""
Entry point for Das et al. (2025) EEG motor imagery replication.

Usage:
  python run_pipeline.py --quick
  python run_pipeline.py --max-subjects 20 --epochs 30
  python run_pipeline.py --gan
"""

import argparse

from das2025_replication.config import DEFAULT_OUTPUT_DIR, DL_EPOCHS
from das2025_replication.run_experiments import run_complete_das2025_replication

if __name__ == "__main__":
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
