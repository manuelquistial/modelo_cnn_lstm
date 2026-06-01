#!/usr/bin/env bash
# Shared helpers for the project .venv (not .env)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

require_venv() {
  if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "ERROR: No existe .venv en $REPO_ROOT" >&2
    echo "       Crea el entorno con: ./scripts/paperspace_setup.sh" >&2
    echo "       (En local: python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .)" >&2
    exit 1
  fi
}

activate_venv() {
  require_venv
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
}

export_repo_env() {
  export MNE_DATA="${MNE_DATA:-$REPO_ROOT/mne_data}"
  export MNE_HOME="${MNE_HOME:-$REPO_ROOT/.mne}"
  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
  export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
}
