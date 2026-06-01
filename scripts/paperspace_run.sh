#!/usr/bin/env bash
# Run Das 2025 replication using .venv/bin/python (after paperspace_setup.sh)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/venv_common.sh"
require_venv
export_repo_env

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
# Full Das et al. (2025): multiclass, trial-wise, 5s→640×2, ICA+CSP, Table 6 epochs, Table 8 binary
if [[ "${PAPER:-0}" == "1" ]]; then
  MODE="multiclass"
  SPLIT="trialwise"
  ARGS=(
    -m das2025_replication.run_experiments
    --mode "$MODE"
    --split "$SPLIT"
    --epochs "$EPOCHS"
    --output-dir "$OUTPUT_DIR"
    --paper-protocol
  )
  if [[ "${GAN:-0}" == "1" ]]; then
    ARGS+=(--gan)
  fi
  if [[ -n "$MAX_SUBJECTS" ]]; then
    ARGS+=(--max-subjects "$MAX_SUBJECTS")
  else
    echo "WARNING: PAPER=1 without MAX_SUBJECTS — all 103 subjects + 6 ROIs (many hours)."
    echo "         Suggested: MAX_SUBJECTS=15 PAPER=1 ./scripts/paperspace_run.sh"
  fi
  if [[ "${VIZ:-0}" != "1" ]]; then
    ARGS+=(--no-viz)
  fi
fi

echo "==> Running: $VENV_PYTHON ${ARGS[*]} ${EXTRA_ARGS[*]}"
"$VENV_PYTHON" "${ARGS[@]}" "${EXTRA_ARGS[@]}"
