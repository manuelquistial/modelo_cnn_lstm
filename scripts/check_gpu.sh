#!/usr/bin/env bash
# Verify TensorFlow sees the NVIDIA GPU on Paperspace
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== nvidia-smi ==="
nvidia-smi || echo "(nvidia-smi not available)"

echo ""
echo "=== TensorFlow ==="
python <<'PY'
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)

if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    with tf.device("/GPU:0"):
        a = tf.random.normal((2000, 2000))
        b = tf.matmul(a, a)
    print("GPU matmul OK — device:", b.device)
else:
    print("No GPU — install: pip install 'tensorflow[and-cuda]>=2.15'")
PY

echo ""
echo "Note: 'cpu_feature_guard' lines at import are normal even when GPU is used."
