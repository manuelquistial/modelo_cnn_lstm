#!/usr/bin/env bash
# Paperspace Gradient / Linux GPU machine — one-time setup
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/venv_common.sh"

echo "==> Repo: $REPO_ROOT"
echo "==> Python: $(python3 --version 2>&1 || true)"

# Prefer Python 3.10–3.12 (TensorFlow)
PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13)' 2>/dev/null; then
  echo "WARNING: Use Python 3.10–3.12. Current: $($PY --version)"
fi

if [[ ! -d .venv ]]; then
  echo "==> Creating .venv ..."
  "$PY" -m venv .venv
fi
require_venv
activate_venv

"$VENV_PIP" install -U pip wheel setuptools

# TensorFlow with GPU on Linux (Paperspace CUDA images)
if command -v nvidia-smi &>/dev/null; then
  echo "==> NVIDIA GPU detected"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  "$VENV_PIP" install "tensorflow[and-cuda]>=2.15" || "$VENV_PIP" install "tensorflow>=2.15"
else
  echo "==> No GPU — installing CPU TensorFlow"
  "$VENV_PIP" install "tensorflow>=2.15"
fi

"$VENV_PIP" install -r requirements.txt
"$VENV_PIP" install -e .

# Persist PhysioNet / MNE data on Paperspace volume (optional)
export MNE_DATA="${MNE_DATA:-$REPO_ROOT/mne_data}"
mkdir -p "$MNE_DATA"
# Non-interactive MNE config (avoid "set as default path [y]/n?" on first download)
export MNE_HOME="${MNE_HOME:-$REPO_ROOT/.mne}"
mkdir -p "$MNE_HOME"
if [[ ! -f "$MNE_HOME/mne-python.json" ]]; then
  printf '%s\n' "{\"eegbci\": {\"path\": \"$MNE_DATA\"}}" > "$MNE_HOME/mne-python.json"
fi
grep -q 'MNE_DATA=' .venv/bin/activate 2>/dev/null || echo "export MNE_DATA=$MNE_DATA" >> .venv/bin/activate
grep -q 'MNE_HOME=' .venv/bin/activate 2>/dev/null || echo "export MNE_HOME=$MNE_HOME" >> .venv/bin/activate

"$VENV_PYTHON" -c "
import tensorflow as tf
print('TensorFlow', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
"

# Jupyter / Paperspace notebook kernel → same .venv
if "$VENV_PYTHON" -c "import ipykernel" 2>/dev/null; then
  "$VENV_PYTHON" -m ipykernel install --user --name=modelo-cnn-lstm --display-name="Python (modelo_cnn_lstm .venv)" || true
fi

echo ""
echo "==> Setup complete (.venv ready)"
echo "    Opción A: source .venv/bin/activate"
echo "    Opción B: .venv/bin/python -m das2025_replication.run_experiments --quick"
echo "    Opción C: QUICK=1 ./scripts/paperspace_run.sh"
echo "    Opción D: make run-quick"
