#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -f outputs/streamlit.pid ]; then
  kill "$(cat outputs/streamlit.pid)" || true
  rm -f outputs/streamlit.pid
fi
