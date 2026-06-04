"""Scheduled and manual GEX refresh: fetch UW and write CSV exports."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

from gex_core.exports import parse_timestamp
from gex_core.history import get_latest_ts

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = [
    item.strip().upper()
    for item in os.environ.get("GEX_DEFAULT_TICKERS", "SPX").split(",")
    if item.strip()
]
DEFAULT_REFRESH_MINUTES = int(os.environ.get("GEX_REFRESH_INTERVAL_MINUTES", "10"))


def is_snapshot_stale(ticker: str, max_age_minutes: int | None = None) -> bool:
    max_age = max_age_minutes if max_age_minutes is not None else DEFAULT_REFRESH_MINUTES
    latest = get_latest_ts(ticker)
    if latest is None:
        return True
    age = datetime.now() - parse_timestamp(latest)
    return age > timedelta(minutes=max_age)


def refresh_ticker(ticker: str, force: bool = False) -> bool:
    """Fetch UW GEX and write a new CSV snapshot. Returns True if a new export was saved."""
    ticker = ticker.upper()
    if not force and not is_snapshot_stale(ticker):
        logger.info("Skipping refresh for %s; latest export is still fresh", ticker)
        return False

    before = get_latest_ts(ticker)

    from main import run

    try:
        run(
            ticker=ticker,
            refresh=True,
            show_plots=False,
            save_plots=False,
            export_csv=True,
            export_dir="data/exports",
        )
    except Exception:
        logger.exception("GEX refresh failed for %s", ticker)
        return False

    after = get_latest_ts(ticker)
    if after is None or (before == after and not force):
        logger.warning("Refresh completed but no new export timestamp for %s", ticker)
        return after is not None and before != after
    logger.info("UW export saved %s for %s", after, ticker)
    return True


def refresh_tickers(tickers: list[str] | None = None, force: bool = False) -> dict[str, bool]:
    symbols = tickers or DEFAULT_TICKERS
    return {symbol.upper(): refresh_ticker(symbol, force=force) for symbol in symbols}
