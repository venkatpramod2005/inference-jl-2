#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p outputs
nohup streamlit run app.py --server.port 7860 --server.address 0.0.0.0 > outputs/streamlit.log 2>&1 &
echo $! > outputs/streamlit.pid
echo "Streamlit started on 0.0.0.0:7860 with PID $(cat outputs/streamlit.pid)"
