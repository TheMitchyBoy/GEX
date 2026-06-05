#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()
PY

_should_backfill() {
  python3 - <<'PY'
import os
from gex_core.storage import count_strike_exports_on_disk

min_rows = int(os.environ.get("GEX_FORECAST_MIN_SNAPSHOTS", "4"))
disk = count_strike_exports_on_disk(os.environ.get("TICKERS", "SPX").split(",")[0].strip() or "SPX")
raise SystemExit(0 if disk < min_rows else 1)
PY
}

if [ "${GEX_STARTUP_BACKFILL:-}" = "1" ] || { [ "${GEX_AUTO_BACKFILL_IF_EMPTY:-1}" = "1" ] && _should_backfill; }; then
  echo "Starting background intraday backfill + model retrain (strike CSV count below minimum)..."
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
