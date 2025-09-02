#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
VENV_DIR=".venv"
REQ_FILE="requirements.txt"
KERNEL_NAME="nprach_synch"
KERNEL_DISPLAY="Python 3.8 (nprach_synch)"

command_exists() { command -v "$1" >/dev/null 2>&1; }

if ! command_exists "$PYTHON_BIN"; then
  echo "[ERROR] python3 not found. Install Python 3.8 (recommended) and retry." >&2
  exit 1
fi

if ! command_exists pip3; then
  echo "[ERROR] pip3 not found. Install pip and retry." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

if [ -f "$REQ_FILE" ]; then
  echo "[INFO] Installing pinned requirements"
  pip install --upgrade --force-reinstall -r "$REQ_FILE"
else
  echo "[WARN] requirements.txt not found; installing core deps"
  pip install --upgrade jupyter matplotlib numpy scipy tensorflow==2.8.4
fi

echo "[INFO] Registering Jupyter kernel: $KERNEL_NAME"
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY" || true

cat << 'EOM'

[OK] Environment is ready.

Shortcuts:
- Run Jupyter Notebook:
    source .venv/bin/activate && jupyter notebook --no-browser --port 8888
  Then open the printed URL in your Windows browser.

- Smoke test (baseline):
    source .venv/bin/activate && python - << 'PY'
from e2e import E2E
BATCH=16
sys = E2E('baseline', False, nprach_num_rep=1, nprach_num_sc=24, fft_size=256, pfa=0.999)
out = sys(BATCH, max_cfo_ppm=10., ue_prob=0.5)
print('OK:', len(out))
PY

Tips (WSL2):
- Increase WSL2 resources via C:\\Users\\<User>\\.wslconfig and run: wsl --shutdown
EOM
