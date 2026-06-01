#!/usr/bin/env bash
# Run Das 2025 replication on Paperspace (after paperspace_setup.sh)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

export MNE_DATA="${MNE_DATA:-$REPO_ROOT/mne_data}"
export PYTHONUNBUFFERED=1

# Limit GPU memory growth (avoid OOM on smaller Paperspace GPUs)
export TF_FORCE_GPU_ALLOW_GROWTH=true

MODE="${MODE:-binary}"
SPLIT="${SPLIT:-subjectwise}"
MAX_SUBJECTS="${MAX_SUBJECTS:-}"
EPOCHS="${EPOCHS:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/das2025_replication}"
EXTRA_ARGS=("$@")

ARGS=(
  -m das2025_replication.run_experiments
  --mode "$MODE"
  --split "$SPLIT"
  --epochs "$EPOCHS"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$MAX_SUBJECTS" ]]; then
  ARGS+=(--max-subjects "$MAX_SUBJECTS")
fi

# Quick smoke: QUICK=1 ./scripts/paperspace_run.sh
if [[ "${QUICK:-0}" == "1" ]]; then
  MAX_SUBJECTS="${MAX_SUBJECTS:-5}"
  ARGS+=(--quick --no-roi --no-segment)
fi

# Paper-like protocol: PAPER=1 ./scripts/paperspace_run.sh
# (multiclass, trial-wise split, 640x2 input, per-ROI epochs from Table 6)
if [[ "${PAPER:-0}" == "1" ]]; then
  MODE="multiclass"
  SPLIT="trialwise"
  ARGS=(
    -m das2025_replication.run_experiments
    --mode "$MODE"
    --split "$SPLIT"
    --epochs "$EPOCHS"
    --output-dir "$OUTPUT_DIR"
    --paper-input
    --paper-roi-epochs
  )
  if [[ -n "$MAX_SUBJECTS" ]]; then
    ARGS+=(--max-subjects "$MAX_SUBJECTS")
  else
    echo "WARNING: PAPER=1 without MAX_SUBJECTS — all 103 subjects + 6 ROIs (many hours)."
    echo "         Suggested: MAX_SUBJECTS=15 PAPER=1 ./scripts/paperspace_run.sh"
  fi
fi

echo "==> Running: python ${ARGS[*]} ${EXTRA_ARGS[*]}"
python "${ARGS[@]}" "${EXTRA_ARGS[@]}"
