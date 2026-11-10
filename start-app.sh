#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
if [[ ! -x .venv/bin/python ]]; then
  echo "Run the one-time setup in app/README.md before starting TrialMatch."
  exit 1
fi
exec .venv/bin/python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
