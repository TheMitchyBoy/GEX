#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()
PY

if [ "${GEX_STARTUP_BACKFILL:-}" = "1" ]; then
  echo "GEX_STARTUP_BACKFILL=1: backfilling exports and retraining in background..."
  (
    python3 scripts/gex_backfill_intraday.py \
      --tickers "${TICKERS:-SPX}" \
      --intraday-days "${GEX_INTRADAY_BACKFILL_DAYS:-90}" \
      --interval-minutes "${GEX_BACKFILL_INTERVAL_MINUTES:-10}"
    python3 scripts/train_gex_model.py \
      --ticker SPX \
      --lookback-days "${GEX_TRAIN_LOOKBACK_DAYS:-90}"
  ) &
fi

exec python3 -m gunicorn \
  --bind "0.0.0.0:${PORT:-8080}" \
  --workers 1 \
  --threads 2 \
  --timeout 120 \
  wsgi:app
