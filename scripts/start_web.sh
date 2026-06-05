#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
from gex_core.env_bootstrap import load_env_files, sync_env_files_from_process

sync_env_files_from_process()
load_env_files()
PY

exec python3 -m gunicorn \
  --bind "0.0.0.0:${PORT:-8080}" \
  --workers 1 \
  --threads 2 \
  --timeout 120 \
  wsgi:app
