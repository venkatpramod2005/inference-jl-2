#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/user/inference-jl-2"
VENV_DIR="${PROJECT_DIR}/.venv"
MINICONDA_DIR="/home/user/miniconda3"
MINICONDA_INSTALLER="/home/user/miniconda.sh"

cd "${PROJECT_DIR}"

echo "== CUDA check =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi was not found. Run this on a CUDA-enabled JarvisLabs GPU instance." >&2
  exit 1
fi
nvidia-smi

echo
echo "== Python environment =="
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    version="$(${candidate} - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
    case "${version}" in
      3.10|3.11|3.12|3.13)
        PYTHON_BIN="${candidate}"
        break
        ;;
    esac
  fi
done

if [ -z "${PYTHON_BIN}" ]; then
  echo "Python 3.10+ not found. Installing Miniconda under /home/user..."
  if [ ! -x "${MINICONDA_DIR}/bin/conda" ]; then
    if command -v curl >/dev/null 2>&1; then
      curl -L -o "${MINICONDA_INSTALLER}" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    elif command -v wget >/dev/null 2>&1; then
      wget -O "${MINICONDA_INSTALLER}" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    else
      echo "ERROR: curl or wget is required to install Miniconda." >&2
      exit 1
    fi
    bash "${MINICONDA_INSTALLER}" -b -p "${MINICONDA_DIR}"
  fi
  "${MINICONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  "${MINICONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
  "${MINICONDA_DIR}/bin/conda" create -y -p "${VENV_DIR}" python=3.10 pip
  # shellcheck source=/dev/null
  source "${MINICONDA_DIR}/bin/activate" "${VENV_DIR}"
else
  echo "Using $(${PYTHON_BIN} --version)"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo
echo "== GPU memory after install =="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv

echo
echo "Setup complete. Activate with: source ${VENV_DIR}/bin/activate"
