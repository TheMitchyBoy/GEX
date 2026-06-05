"""Operational health payload for dashboards and ``/health``."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from gex_core.env_bootstrap import uw_api_key_diagnostics
from gex_core.exports import EXPORT_DIR, list_export_timestamps
from gex_core.history import build_history, get_latest_ts
from gex_core.models_manifest import load_manifest
from gex_core.predict import MIN_OVERLAY_TRAIN_ROWS
from gex_core.refresh import DEFAULT_REFRESH_MINUTES
from gex_core.export_diagnostics import summarize_export_state
from gex_core.storage import db_path, export_age_minutes
from gex_core.tickers import PRIMARY_TICKER


def build_system_status(ticker: str | None = None) -> dict:
    ticker = (ticker or PRIMARY_TICKER).upper()
    latest_ts = get_latest_ts(ticker, EXPORT_DIR)
    age_min = export_age_minutes(ticker, EXPORT_DIR)
    history = build_history(ticker, EXPORT_DIR)
    manifest = load_manifest(ticker)
    n_train = None
    if manifest:
        n_train = (manifest.get("metrics") or {}).get("n_train")

    scheduler_disabled = os.environ.get("GEX_DISABLE_SCHEDULER", "").lower() in {"1", "true", "yes"}

    stale = False
    if age_min is not None and age_min > DEFAULT_REFRESH_MINUTES * 3:
        stale = True

    export_state: dict = {}
    try:
        export_state = summarize_export_state(ticker)
    except Exception:
        pass

    return {
        "ticker": ticker,
        "healthy": bool(history) and not stale,
        **uw_api_key_diagnostics(),
        "scheduler_enabled": not scheduler_disabled,
        "latest_export_ts": latest_ts,
        "export_age_minutes": age_min,
        "export_stale": stale,
        "history_depth": export_state.get("catalog_timestamps", len(list_export_timestamps(ticker, EXPORT_DIR))),
        "history_loaded": len(history),
        "strike_csv_on_disk": export_state.get("strike_csv_on_disk"),
        "forecast_loadable": export_state.get("forecast_loadable"),
        "export_index_stale": export_state.get("index_stale"),
        "export_needs_backfill": export_state.get("needs_backfill"),
        "index_db_present": db_path().exists(),
        "refresh_interval_minutes": DEFAULT_REFRESH_MINUTES,
        "model_overlay_active": n_train is not None and n_train >= MIN_OVERLAY_TRAIN_ROWS,
        "model_training_rows": n_train,
        "model_overlay_min_rows": MIN_OVERLAY_TRAIN_ROWS,
        "alert_webhook_configured": bool(os.environ.get("GEX_ALERT_WEBHOOK_URL")),
        "alert_auto_dispatch": os.environ.get("GEX_ALERT_AUTO_DISPATCH", "").lower() in {"1", "true", "yes"},
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
