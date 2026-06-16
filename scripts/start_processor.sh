#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()
PY

python3 scripts/init_postgres_schema.py

if [ "${GEX_MIGRATE_SQLITE:-}" != "0" ]; then
  python3 scripts/migrate_sqlite_to_postgres.py --if-needed || true
fi

# Processor mode: Postgres is canonical; skip CSV unless explicitly enabled.
export GEX_EXPORT_CSV="${GEX_EXPORT_CSV:-0}"

exec python3 -m gex_core.processor
