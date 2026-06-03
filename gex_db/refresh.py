from __future__ import annotations

import logging
import os
from datetime import datetime

from gex_db.store import get_latest_ts, is_snapshot_stale, save_snapshot

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = [
    item.strip().upper()
    for item in os.environ.get("GEX_DEFAULT_TICKERS", "SPX").split(",")
    if item.strip()
]
DEFAULT_REFRESH_MINUTES = int(os.environ.get("GEX_REFRESH_INTERVAL_MINUTES", "10"))


def refresh_ticker(ticker: str, force: bool = False) -> bool:
    """
    Run GEX analysis for a ticker and persist results to the database.

    Returns True when a new snapshot was saved.
    """
    ticker = ticker.upper()
    if not force and not is_snapshot_stale(ticker, max_age_minutes=DEFAULT_REFRESH_MINUTES):
        logger.info("Skipping refresh for %s; latest snapshot is still fresh", ticker)
        return False

    # Import lazily to avoid circular imports with web_app.
    from main import run

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
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

    latest = get_latest_ts(ticker)
    if latest is None:
        logger.warning("Refresh completed but no snapshot found for %s", ticker)
        return False

    logger.info("GEX refresh saved snapshot %s for %s", latest, ticker)
    return True


def refresh_tickers(tickers: list[str] | None = None, force: bool = False) -> dict[str, bool]:
    symbols = tickers or DEFAULT_TICKERS
    return {symbol.upper(): refresh_ticker(symbol, force=force) for symbol in symbols}
