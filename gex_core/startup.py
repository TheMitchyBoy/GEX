"""Deferred process startup so gunicorn can bind before heavy background work."""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_started = False
_lock = threading.Lock()


def deferred_web_startup(
    *,
    refresh_fn,
    price_stream_fn,
    delay_seconds: float | None = None,
) -> None:
    """Start scheduler + UW price websocket after a short delay (once per worker)."""
    global _started
    with _lock:
        if _started:
            return
        _started = True

    if delay_seconds is None:
        try:
            delay_seconds = float(os.environ.get("GEX_DEFERRED_STARTUP_SEC", "3"))
        except (TypeError, ValueError):
            delay_seconds = 3.0

    def _run() -> None:
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        try:
            refresh_fn()
        except Exception:
            logger.exception("Background refresh scheduler failed to start")
        try:
            price_stream_fn()
        except Exception:
            logger.exception("UW price websocket failed to start")

    threading.Thread(target=_run, name="gex-deferred-startup", daemon=True).start()


def should_retrain_on_start(ticker: str = "SPX") -> bool:
    """Return True when a background retrain is warranted at container boot."""
    if os.environ.get("GEX_RETRAIN_ON_START", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False

    from gex_core.models_manifest import load_manifest
    from gex_core.storage import count_strike_exports_on_disk

    ticker = ticker.upper()
    disk = count_strike_exports_on_disk(ticker)
    if disk < 4:
        return False

    manifest = load_manifest(ticker)
    if not manifest:
        logger.info("Retrain on start: no manifest for %s (%s CSVs on disk)", ticker, disk)
        return True

    catalog = int(manifest.get("catalog_timestamps") or manifest.get("n_snapshots") or 0)
    lookback = manifest.get("lookback_days")
    configured = int(os.environ.get("GEX_TRAIN_LOOKBACK_DAYS", "0"))
    if lookback is not None and int(lookback) > 0 and configured == 0:
        logger.info(
            "Retrain on start: manifest used %s-day window; disk has %s CSVs (configured full catalog)",
            lookback,
            disk,
        )
        return True

    gap = max(50, int(disk * 0.02))
    if disk > catalog + gap:
        logger.info(
            "Retrain on start: disk=%s catalog_in_manifest=%s (gap %s)",
            disk,
            catalog,
            gap,
        )
        return True

    logger.info("Retrain on start skipped: manifest already covers %s/%s CSVs", catalog, disk)
    return False
