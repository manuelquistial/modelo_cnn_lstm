#!/usr/bin/env bash
# Paperspace Gradient / Linux GPU machine — one-time setup
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Repo: $REPO_ROOT"
echo "==> Python: $(python3 --version 2>&1 || true)"

# Prefer Python 3.10–3.12 (TensorFlow)
PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13)' 2>/dev/null; then
  echo "WARNING: Use Python 3.10–3.12. Current: $($PY --version)"
fi

if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -U pip wheel setuptools

# TensorFlow with GPU on Linux (Paperspace CUDA images)
if command -v nvidia-smi &>/dev/null; then
  echo "==> NVIDIA GPU detected"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  pip install "tensorflow[and-cuda]>=2.15" || pip install "tensorflow>=2.15"
else
  echo "==> No GPU — installing CPU TensorFlow"
  pip install "tensorflow>=2.15"
fi

pip install -r requirements.txt
pip install -e .

# Persist PhysioNet / MNE data on Paperspace volume (optional)
export MNE_DATA="${MNE_DATA:-$REPO_ROOT/mne_data}"
mkdir -p "$MNE_DATA"
echo "export MNE_DATA=$MNE_DATA" >> .venv/bin/activate

python -c "
import tensorflow as tf
print('TensorFlow', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
"

echo "==> Setup complete. Activate: source .venv/bin/activate"
