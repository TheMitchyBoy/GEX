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


def local_export_strike_count(export_dir: Path | None = None) -> int:
    from gex_core.exports import EXPORT_DIR, scan_export_timestamps

    export_dir = export_dir or EXPORT_DIR
    ticker = os.environ.get("GEX_DEFAULT_TICKERS", "SPX").split(",")[0].strip().upper() or "SPX"
    return len(scan_export_timestamps(ticker, export_dir))


def bootstrap_postgres_data(ticker: str | None = None) -> dict[str, object]:
    """Import on-disk exports and/or UW backfill when Postgres history is sparse."""
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
    }

    if not use_postgres():
        report["reason"] = "DATABASE_URL not set"
        return report

    ensure_postgres_schema()
    report["snapshot_count_before"] = count_snapshots(ticker)

    csv_count = local_export_strike_count(EXPORT_DIR)
    if csv_count > 0 and os.environ.get("GEX_IMPORT_EXPORTS_ON_START", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        try:
            from gex_core.import_exports import import_ticker_exports, summarize_import_results

            results = import_ticker_exports(ticker, skip_existing=True, force=False)
            counts = summarize_import_results(results)
            report["imported"] = counts.get("imported", 0)
            logger.info(
                "CSV import for %s: imported=%s skipped=%s errors=%s",
                ticker,
                counts.get("imported"),
                counts.get("skipped"),
                counts.get("errors"),
            )
        except Exception:
            logger.exception("CSV import bootstrap failed for %s", ticker)

    report["snapshot_count_after"] = count_snapshots(ticker)
    if report["snapshot_count_after"] >= backfill_min_snapshots():
        report["skipped"] = True
        report["reason"] = "enough_snapshots"
        return report

    if os.environ.get("GEX_STARTUP_BACKFILL", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        report["skipped"] = True
        report["reason"] = "GEX_STARTUP_BACKFILL disabled"
        return report

    from gex_core.intraday_backfill import backfill_recent_daily, backfill_recent_intraday

    os.environ.setdefault("GEX_BACKFILL_MODE", "1")
    os.environ.setdefault("GEX_HARD_REJECT_TOTAL_GEX_MISMATCH", "0")
    os.environ.setdefault("GEX_MIN_STRIKE_COUNT", "3")

    intraday_days = int(os.environ.get("GEX_INTRADAY_BACKFILL_DAYS", "90"))
    daily_days = int(os.environ.get("GEX_DAILY_BACKFILL_DAYS", "90"))
    interval = int(os.environ.get("GEX_BACKFILL_INTERVAL_MINUTES", "10"))
    report["backfill_started"] = True

    try:
        intraday = backfill_recent_intraday(
            ticker,
            days=intraday_days,
            interval_minutes=interval,
            since_date="",
        )
        daily = backfill_recent_daily(ticker, days=daily_days)
        report["intraday_saved"] = sum(intraday.values())
        report["daily_saved"] = sum(1 for value in daily.values() if value)
    except Exception as exc:
        logger.exception("UW backfill bootstrap failed for %s", ticker)
        report["reason"] = str(exc)

    report["snapshot_count_after"] = count_snapshots(ticker)
    return report
