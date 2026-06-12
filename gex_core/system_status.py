"""Operational health payload for dashboards and ``/health``."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HEALTH_CACHE: dict[str, tuple[float, dict]] = {}

from gex_core.env_bootstrap import uw_api_key_diagnostics
from gex_core.exports import EXPORT_DIR, filter_export_timestamps, list_export_timestamps, parse_timestamp
from gex_core.history import get_latest_ts
from gex_core.models_manifest import load_manifest
from gex_core.predict import MIN_OVERLAY_TRAIN_ROWS
from gex_core.refresh import DEFAULT_REFRESH_MINUTES
from gex_core.storage import count_strike_exports_on_disk, db_path, export_age_minutes, sync_ticker_exports
from gex_core.tickers import PRIMARY_TICKER


def _health_cache_ttl() -> float:
    try:
        return max(5.0, float(os.environ.get("GEX_HEALTH_CACHE_SEC", "30")))
    except (TypeError, ValueError):
        return 30.0


def build_system_status(
    ticker: str | None = None,
    *,
    use_cache: bool = True,
    light: bool = False,
) -> dict:
    ticker = (ticker or PRIMARY_TICKER).upper()
    cache_key = f"{ticker}:{'light' if light else 'full'}"
    if use_cache:
        cached = _HEALTH_CACHE.get(cache_key)
        if cached and (time.monotonic() - cached[0]) < _health_cache_ttl():
            return dict(cached[1])
    latest_ts = get_latest_ts(ticker, EXPORT_DIR)
    age_min = export_age_minutes(ticker, EXPORT_DIR)
    if light:
        try:
            from gex_core.storage import list_indexed_timestamps

            timestamps = list_indexed_timestamps(ticker)
        except Exception:
            timestamps = []
        if not timestamps and latest_ts:
            timestamps = [latest_ts]
    else:
        timestamps = list_export_timestamps(ticker, EXPORT_DIR)
    history_loaded = 0
    if timestamps:
        latest_dt = parse_timestamp(timestamps[-1])
        since = latest_dt - timedelta(days=90)
        history_loaded = len(
            filter_export_timestamps(timestamps, since=since, max_timestamps=240)
        )
    manifest = load_manifest(ticker)
    n_train = None
    if manifest:
        n_train = (manifest.get("metrics") or {}).get("n_train")

    scheduler_disabled = os.environ.get("GEX_DISABLE_SCHEDULER", "").lower() in {"1", "true", "yes"}

    stale = False
    if age_min is not None and age_min > DEFAULT_REFRESH_MINUTES * 3:
        stale = True

    strike_csv_on_disk = None
    catalog_timestamps = len(timestamps)
    if not light:
        catalog_timestamps = len(list_export_timestamps(ticker, EXPORT_DIR))
        try:
            sync_ticker_exports(ticker, EXPORT_DIR)
            strike_csv_on_disk = count_strike_exports_on_disk(ticker, EXPORT_DIR)
            catalog_timestamps = len(list_export_timestamps(ticker, EXPORT_DIR))
        except Exception:
            pass

    ready = history_loaded > 0
    payload = {
        "ticker": ticker,
        "ready": ready,
        "healthy": ready and not stale,
        **uw_api_key_diagnostics(),
        "scheduler_enabled": not scheduler_disabled,
        "latest_export_ts": latest_ts,
        "export_age_minutes": age_min,
        "export_stale": stale,
        "history_depth": catalog_timestamps,
        "history_loaded": history_loaded,
        "strike_csv_on_disk": strike_csv_on_disk,
        "forecast_loadable": history_loaded,
        "export_index_stale": (
            strike_csv_on_disk is not None and catalog_timestamps > strike_csv_on_disk
        ),
        "export_needs_backfill": (
            strike_csv_on_disk is not None and strike_csv_on_disk < 4
        ),
        "index_db_present": db_path().exists(),
        "refresh_interval_minutes": DEFAULT_REFRESH_MINUTES,
        "trader_cycle_seconds": int(os.environ.get("GEX_TRADER_CYCLE_SECONDS", "30")),
        "auto_trader_enabled": os.environ.get("GEX_AUTO_TRADER", "").strip().lower() in {"1", "true", "yes"},
        "model_overlay_active": n_train is not None and n_train >= MIN_OVERLAY_TRAIN_ROWS,
        "model_training_rows": n_train,
        "model_overlay_min_rows": MIN_OVERLAY_TRAIN_ROWS,
        "alert_webhook_configured": bool(os.environ.get("GEX_ALERT_WEBHOOK_URL")),
        "alert_auto_dispatch": os.environ.get("GEX_ALERT_AUTO_DISPATCH", "").lower() in {"1", "true", "yes"},
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if use_cache:
        _HEALTH_CACHE[cache_key] = (time.monotonic(), dict(payload))
    return payload
