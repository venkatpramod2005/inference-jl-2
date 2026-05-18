#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p outputs
STREAMLIT_PORT="${STREAMLIT_PORT:-6006}"
nohup streamlit run app.py --server.port "${STREAMLIT_PORT}" --server.address 0.0.0.0 > outputs/streamlit.log 2>&1 &
echo $! > outputs/streamlit.pid
echo "Streamlit started on 0.0.0.0:${STREAMLIT_PORT} with PID $(cat outputs/streamlit.pid)"
