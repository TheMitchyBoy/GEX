#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Processor mode must be set before schema init (processor-only tables).
export GEX_PROCESSOR_MODE=1
export GEX_EXPORT_CSV="${GEX_EXPORT_CSV:-0}"
export GEX_SUMMARY_MARKET_FEATURES="${GEX_SUMMARY_MARKET_FEATURES:-0}"
export GEX_MIGRATE_SQLITE="${GEX_MIGRATE_SQLITE:-0}"

python3 - <<'PY'
from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()
PY

python3 scripts/init_postgres_schema.py

if [ "${GEX_MIGRATE_SQLITE}" != "0" ]; then
  python3 scripts/migrate_sqlite_to_postgres.py --if-needed || true
fi

exec python3 -m gex_core.processor
