#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs
STREAMLIT_PORT="${STREAMLIT_PORT:-6006}"
STREAMLIT_BIN="${STREAMLIT_BIN:-.venv/bin/streamlit}"
nohup "${STREAMLIT_BIN}" run app.py --server.port "${STREAMLIT_PORT}" --server.address 0.0.0.0 > outputs/streamlit.log 2>&1 &
echo $! > outputs/streamlit.pid
echo "Streamlit started on 0.0.0.0:${STREAMLIT_PORT} with PID $(cat outputs/streamlit.pid)"
