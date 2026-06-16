#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Processor mode must be set before schema init (processor-only tables).
export GEX_PROCESSOR_MODE=1
export GEX_EXPORT_CSV="${GEX_EXPORT_CSV:-0}"
export GEX_SUMMARY_MARKET_FEATURES="${GEX_SUMMARY_MARKET_FEATURES:-0}"
export GEX_LIGHTWEIGHT_MARKET_CONTEXT="${GEX_LIGHTWEIGHT_MARKET_CONTEXT:-1}"
export GEX_MIGRATE_SQLITE="${GEX_MIGRATE_SQLITE:-0}"
export GEX_SKIP_DUPLICATE_SNAPSHOTS="${GEX_SKIP_DUPLICATE_SNAPSHOTS:-1}"
export GEX_RECONCILE_PREDICTIONS="${GEX_RECONCILE_PREDICTIONS:-1}"
export GEX_HARD_REJECT_TOTAL_GEX_MISMATCH="${GEX_HARD_REJECT_TOTAL_GEX_MISMATCH:-1}"
export GEX_QUALITY_ALERTS="${GEX_QUALITY_ALERTS:-0}"
export GEX_MAX_DATA_LAG_SEC="${GEX_MAX_DATA_LAG_SEC:-1200}"
export GEX_SPOT_DISAGREEMENT_TOLERANCE_PCT="${GEX_SPOT_DISAGREEMENT_TOLERANCE_PCT:-0.005}"
export GEX_MIN_STRIKE_GEX_BN="${GEX_MIN_STRIKE_GEX_BN:-1e-6}"

python3 - <<'PY'
from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()
PY

python3 scripts/init_postgres_schema.py

if [ "${GEX_MIGRATE_SQLITE}" != "0" ]; then
  python3 scripts/migrate_sqlite_to_postgres.py --if-needed || true
fi

if [ "${GEX_IMPORT_EXPORTS_ON_START:-0}" = "1" ]; then
  python3 scripts/import_exports_to_postgres.py --if-missing || true
fi

if [ "${GEX_STARTUP_BACKFILL:-0}" = "1" ]; then
  python3 scripts/backfill_postgres_history.py --if-sparse || true
fi

exec python3 -m gex_core.processor
