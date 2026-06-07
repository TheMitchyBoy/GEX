"""Scheduled and manual GEX refresh: fetch UW and write CSV exports.

The refresh loop is file-based — each successful run appends a timestamped
snapshot under ``data/exports/``. ``is_snapshot_stale`` compares the latest
export age to ``GEX_REFRESH_INTERVAL_MINUTES`` (default 2 minutes for the web
dashboard). ``refresh_recent_tickers`` backfills one snapshot per weekday over
a lookback window for prediction history.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from gex_core.env_bootstrap import parse_env_minutes
from gex_core.exports import parse_timestamp
from gex_core.history import get_latest_ts, list_timestamps
from gex_core.tickers import PRIMARY_TICKER, SUPPORTED_TICKERS, is_supported_ticker

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = list(SUPPORTED_TICKERS)
DEFAULT_REFRESH_MINUTES = parse_env_minutes("GEX_REFRESH_INTERVAL_MINUTES", 2.0)


def is_snapshot_stale(ticker: str, max_age_minutes: float | None = None) -> bool:
    max_age = max_age_minutes if max_age_minutes is not None else DEFAULT_REFRESH_MINUTES
    latest = get_latest_ts(ticker)
    if latest is None:
        return True
    age = datetime.now() - parse_timestamp(latest)
    return age > timedelta(minutes=max_age)


def has_snapshot_for_market_date(ticker: str, market_date: str) -> bool:
    """Return True when any export snapshot already exists for YYYY-MM-DD."""
    prefix = f"{market_date}_"
    return any(ts.startswith(prefix) for ts in list_timestamps(ticker))


def refresh_ticker(
    ticker: str,
    force: bool = False,
    market_date: str | None = None,
    *,
    intraday: bool = False,
) -> bool:
    """Fetch UW GEX and write a new CSV snapshot. Returns True if a new export was saved."""
    ticker = ticker.upper()
    if not is_supported_ticker(ticker):
        logger.warning("Skipping refresh for unsupported ticker %s; dashboard is SPX-only", ticker)
        return False
    if market_date and not intraday and not force and has_snapshot_for_market_date(ticker, market_date):
        logger.info("Skipping refresh for %s on %s; export already exists", ticker, market_date)
        return True
    if not market_date and not force and not is_snapshot_stale(ticker):
        logger.info("Skipping refresh for %s; latest export is still fresh", ticker)
        return False

    before = get_latest_ts(ticker)

    try:
        if market_date and not intraday:
            from main import run

            run(
                ticker=ticker,
                refresh=True,
                show_plots=False,
                save_plots=False,
                export_csv=True,
                export_dir="data/exports",
                market_date=market_date,
            )
        else:
            from gex_core.intraday_backfill import export_live_strike_snapshot

            ts = export_live_strike_snapshot(ticker, force=force)
            if ts is None:
                return False
    except Exception:
        logger.exception("GEX refresh failed for %s", ticker)
        return False

    try:
        from gex_core.backtest_metrics import clear_cache as clear_backtest_cache
        from gex_core.history import clear_history_cache

        clear_history_cache()
        clear_backtest_cache()
    except Exception:
        pass

    after = get_latest_ts(ticker)
    if market_date:
        saved = has_snapshot_for_market_date(ticker, market_date)
        if saved:
            logger.info("UW export saved for %s on %s", ticker, market_date)
        else:
            logger.warning("Refresh completed but no export found for %s on %s", ticker, market_date)
        return saved
    if after is None or (before == after and not force):
        logger.warning("Refresh completed but no new export timestamp for %s", ticker)
        return after is not None and before != after
    logger.info("UW export saved %s for %s", after, ticker)
    return True


def refresh_tickers(
    tickers: list[str] | None = None,
    force: bool = False,
    market_date: str | None = None,
) -> dict[str, bool]:
    symbols = [symbol.upper() for symbol in (tickers or DEFAULT_TICKERS) if is_supported_ticker(symbol)]
    if not symbols:
        symbols = [PRIMARY_TICKER]
    return {symbol.upper(): refresh_ticker(symbol, force=force, market_date=market_date) for symbol in symbols}


def recent_market_dates(days: int = 7, today: date | None = None) -> list[str]:
    """Return weekday date strings from oldest to newest over the recent calendar window."""
    anchor = today or date.today()
    days = max(1, days)
    return [
        (anchor - timedelta(days=offset)).isoformat()
        for offset in range(days - 1, -1, -1)
        if (anchor - timedelta(days=offset)).weekday() < 5
    ]


def refresh_recent_tickers(
    tickers: list[str] | None = None,
    *,
    days: int = 7,
    force: bool = False,
) -> dict[str, dict[str, bool]]:
    """Backfill one UW snapshot per calendar date in the recent lookback window."""
    results: dict[str, dict[str, bool]] = {}
    for market_date in recent_market_dates(days=days):
        results[market_date] = refresh_tickers(tickers, force=force, market_date=market_date)
    return results
