#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()
PY

# Keep HTML dashboard responsive on small Railway instances.
export GEX_BACKTEST_METRICS="${GEX_BACKTEST_METRICS:-0}"
export GEX_DASHBOARD_SKIP_BACKTEST="${GEX_DASHBOARD_SKIP_BACKTEST:-1}"
export GEX_PAGE_MINIMAL_LOAD="${GEX_PAGE_MINIMAL_LOAD:-1}"
export GEX_PAGE_UW_PEEK_ONLY="${GEX_PAGE_UW_PEEK_ONLY:-1}"
export GEX_UW_FETCH_TIMEOUT_SEC="${GEX_UW_FETCH_TIMEOUT_SEC:-4}"
export GEX_AGENT_FETCH_EXTRAS="${GEX_AGENT_FETCH_EXTRAS:-0}"
export GEX_AUTO_BACKFILL_IF_EMPTY="${GEX_AUTO_BACKFILL_IF_EMPTY:-0}"
export GEX_DAILY_LEARNING="${GEX_DAILY_LEARNING:-0}"
export GEX_WALL_GEX_AUTO="${GEX_WALL_GEX_AUTO:-0}"
export GEX_WALL_GEX_CYCLE_SECONDS="${GEX_WALL_GEX_CYCLE_SECONDS:-120}"
export GEX_REFRESH_INTERVAL_MINUTES="${GEX_REFRESH_INTERVAL_MINUTES:-10}"
export GEX_DASHBOARD_HISTORY_MAX="${GEX_DASHBOARD_HISTORY_MAX:-24}"
export GEX_PREDICTION_HISTORY_MAX="${GEX_PREDICTION_HISTORY_MAX:-24}"
export GEX_WALL_GEX_LIVE_CACHE_SEC="${GEX_WALL_GEX_LIVE_CACHE_SEC:-60}"
export GEX_STRATEGY_POLL_SEC="${GEX_STRATEGY_POLL_SEC:-120}"
export GEX_WALL_GEX_STATUS_POLL_SEC="${GEX_WALL_GEX_STATUS_POLL_SEC:-90}"

_should_backfill() {
  python3 - <<'PY'
import os
from gex_core.storage import list_indexed_timestamps

ticker = os.environ.get("TICKERS", "SPX").split(",")[0].strip() or "SPX"
min_rows = int(os.environ.get("GEX_FORECAST_MIN_SNAPSHOTS", "4"))
try:
    indexed = len(list_indexed_timestamps(ticker))
except Exception:
    indexed = 0
raise SystemExit(0 if indexed < min_rows else 1)
PY
}

_run_background_train() {
  if command -v nice >/dev/null 2>&1; then
    nice -n 10 python3 scripts/train_gex_model.py \
      --ticker SPX \
      --lookback-days "${GEX_TRAIN_LOOKBACK_DAYS:-0}"
  else
    python3 scripts/train_gex_model.py \
      --ticker SPX \
      --lookback-days "${GEX_TRAIN_LOOKBACK_DAYS:-0}"
  fi
}

if [ "${GEX_STARTUP_BACKFILL:-}" = "1" ] || { [ "${GEX_AUTO_BACKFILL_IF_EMPTY:-1}" = "1" ] && _should_backfill; }; then
  echo "Starting background intraday backfill + model retrain (strike CSV count below minimum)..."
  (
    python3 scripts/gex_backfill_intraday.py \
      --tickers "${TICKERS:-SPX}" \
      --intraday-days "${GEX_INTRADAY_BACKFILL_DAYS:-90}" \
      --daily-days "${GEX_DAILY_BACKFILL_DAYS:-90}" \
      --interval-minutes "${GEX_BACKFILL_INTERVAL_MINUTES:-10}" || true
    _run_background_train || true
  ) &
fi

if python3 - <<'PY'
from gex_core.startup import should_retrain_on_start
import os
ticker = os.environ.get("TICKERS", "SPX").split(",")[0].strip() or "SPX"
raise SystemExit(0 if should_retrain_on_start(ticker) else 1)
PY
then
  echo "Starting background full-catalog model retrain (manifest stale vs disk)..."
  ( _run_background_train || true ) &
fi

exec python3 -m gunicorn \
  --bind "0.0.0.0:${PORT:-8080}" \
  --workers 1 \
  --threads 4 \
  --timeout 180 \
  --graceful-timeout 30 \
  wsgi:app
