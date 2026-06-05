#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()
PY

exec python3 -m gunicorn \
  --bind "0.0.0.0:${PORT:-8080}" \
  --workers 1 \
  --threads 2 \
  --timeout 120 \
  wsgi:app
