"""Bootstrap PostgreSQL snapshot history from local CSVs and/or UW API."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def backfill_min_snapshots() -> int:
    try:
        return int(os.environ.get("GEX_BACKFILL_MIN_SNAPSHOTS", "30"))
    except (TypeError, ValueError):
        return 30


def needs_postgres_bootstrap(ticker: str) -> bool:
    from gex_core.db import use_postgres
    from gex_core.storage import count_snapshots

    if not use_postgres():
        return False
    return count_snapshots(ticker.upper()) < backfill_min_snapshots()


def postgres_latest_market_date(ticker: str) -> str | None:
    from gex_core.storage import latest_timestamp

    latest = latest_timestamp(ticker.upper())
    if not latest:
        return None
    return latest.split("_", 1)[0]


def needs_postgres_catchup(ticker: str) -> bool:
    """True when Postgres latest snapshot is before today's market calendar date."""
    from gex_core.db import use_postgres
    from gex_core.market_time import market_today

    if not use_postgres():
        return False
    latest_day = postgres_latest_market_date(ticker)
    if not latest_day:
        return True
    return latest_day < market_today()


def local_export_strike_count(export_dir: Path | None = None) -> int:
    from gex_core.exports import EXPORT_DIR, scan_export_timestamps

    export_dir = export_dir or EXPORT_DIR
    ticker = os.environ.get("GEX_DEFAULT_TICKERS", "SPX").split(",")[0].strip().upper() or "SPX"
    return len(scan_export_timestamps(ticker, export_dir))


def _import_local_exports(ticker: str, export_dir: Path) -> int:
    if local_export_strike_count(export_dir) <= 0:
        return 0
    if os.environ.get("GEX_IMPORT_EXPORTS_ON_START", "1").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return 0
    try:
        from gex_core.import_exports import import_ticker_exports, summarize_import_results

        results = import_ticker_exports(ticker, export_dir=export_dir, skip_existing=True, force=False)
        counts = summarize_import_results(results)
        logger.info(
            "CSV import for %s: imported=%s skipped=%s errors=%s",
            ticker,
            counts.get("imported"),
            counts.get("skipped"),
            counts.get("errors"),
        )
        return int(counts.get("imported", 0))
    except Exception:
        logger.exception("CSV import failed for %s", ticker)
        return 0


def _configure_backfill_env() -> None:
    os.environ.setdefault("GEX_BACKFILL_MODE", "1")
    os.environ.setdefault("GEX_HARD_REJECT_TOTAL_GEX_MISMATCH", "0")
    os.environ.setdefault("GEX_MIN_STRIKE_COUNT", "3")


def _run_uw_backfill(
    ticker: str,
    *,
    since_date: str | None,
    intraday_days: int | None = None,
    daily_days: int | None = None,
    interval_minutes: int | None = None,
) -> dict[str, object]:
    from gex_core.intraday_backfill import backfill_recent_daily, backfill_recent_intraday

    intraday_days = intraday_days if intraday_days is not None else int(
        os.environ.get("GEX_INTRADAY_BACKFILL_DAYS", "90")
    )
    daily_days = daily_days if daily_days is not None else int(os.environ.get("GEX_DAILY_BACKFILL_DAYS", "90"))
    interval = interval_minutes if interval_minutes is not None else int(
        os.environ.get("GEX_BACKFILL_INTERVAL_MINUTES", "10")
    )
    since = since_date if since_date is not None else postgres_latest_market_date(ticker)

    intraday = backfill_recent_intraday(
        ticker,
        days=intraday_days,
        interval_minutes=interval,
        since_date=since or "",
    )
    daily = backfill_recent_daily(ticker, days=daily_days)
    return {
        "since_date": since,
        "intraday_saved": sum(intraday.values()),
        "daily_saved": sum(1 for value in daily.values() if value),
        "intraday_days": intraday_days,
        "daily_days": daily_days,
        "interval_minutes": interval,
    }


def sync_postgres_snapshots(
    ticker: str | None = None,
    *,
    force_backfill: bool = False,
) -> dict[str, object]:
    """Import on-disk exports and backfill UW history through the latest trading day."""
    from gex_core.db import ensure_postgres_schema, use_postgres
    from gex_core.exports import EXPORT_DIR
    from gex_core.storage import count_snapshots
    from gex_core.tickers import PRIMARY_TICKER

    ticker = (ticker or os.environ.get("GEX_DEFAULT_TICKERS", PRIMARY_TICKER).split(",")[0]).strip().upper()
    report: dict[str, object] = {
        "ticker": ticker,
        "postgres": use_postgres(),
        "snapshot_count_before": 0,
        "snapshot_count_after": 0,
        "imported": 0,
        "backfill_started": False,
        "skipped": False,
        "reason": None,
        "latest_market_date_before": None,
        "latest_market_date_after": None,
    }

    if not use_postgres():
        report["reason"] = "DATABASE_URL not set"
        return report

    ensure_postgres_schema()
    report["snapshot_count_before"] = count_snapshots(ticker)
    report["latest_market_date_before"] = postgres_latest_market_date(ticker)

    report["imported"] = _import_local_exports(ticker, EXPORT_DIR)
    report["snapshot_count_after"] = count_snapshots(ticker)
    report["latest_market_date_after"] = postgres_latest_market_date(ticker)

    sparse = report["snapshot_count_after"] < backfill_min_snapshots()
    stale = needs_postgres_catchup(ticker)
    if not force_backfill and not sparse and not stale:
        report["skipped"] = True
        report["reason"] = "up_to_date"
        return report

    if os.environ.get("GEX_STARTUP_BACKFILL", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        report["skipped"] = True
        report["reason"] = "GEX_STARTUP_BACKFILL disabled"
        return report

    _configure_backfill_env()
    report["backfill_started"] = True

    try:
        backfill_report = _run_uw_backfill(
            ticker,
            since_date=report["latest_market_date_after"] or report["latest_market_date_before"],
        )
        report.update(backfill_report)
    except Exception as exc:
        logger.exception("UW backfill sync failed for %s", ticker)
        report["reason"] = str(exc)

    report["imported"] = int(report["imported"]) + _import_local_exports(ticker, EXPORT_DIR)
    report["snapshot_count_after"] = count_snapshots(ticker)
    report["latest_market_date_after"] = postgres_latest_market_date(ticker)
    return report


def bootstrap_postgres_data(ticker: str | None = None) -> dict[str, object]:
    """Import on-disk exports and/or UW backfill when Postgres history is sparse or stale."""
    return sync_postgres_snapshots(ticker)
