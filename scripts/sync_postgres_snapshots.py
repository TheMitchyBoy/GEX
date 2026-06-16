#!/usr/bin/env python3
"""Catch up PostgreSQL snapshot tables from local CSV exports and UW API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()

from gex_core.bootstrap_data import (
    missing_market_dates,
    needs_postgres_catchup,
    postgres_latest_market_date,
    sync_postgres_snapshots,
)
from gex_core.data_root import configure_data_paths
from gex_core.db import ensure_postgres_schema, use_postgres
from gex_core.storage import count_snapshots, latest_timestamp
from gex_core.tickers import SUPPORTED_TICKERS


def _parse_tickers(raw: str | None) -> list[str]:
    if not raw:
        return list(SUPPORTED_TICKERS)
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", help="Comma-separated tickers (default: all supported)")
    parser.add_argument(
        "--force-backfill",
        action="store_true",
        help="Run UW backfill even when Postgres already has today's market date",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Lookback window for missing-day detection (default: GEX_CATCHUP_LOOKBACK_DAYS)",
    )
    args = parser.parse_args()

    configure_data_paths()

    if not use_postgres():
        print("DATABASE_URL is not set — cannot sync into PostgreSQL.")
        return 1

    ensure_postgres_schema()
    tickers = _parse_tickers(args.tickers)
    ok = True

    for ticker in tickers:
        before = count_snapshots(ticker)
        latest_before = latest_timestamp(ticker)
        print(f"{ticker}: {before} snapshots, latest={latest_before or 'none'}")
        missing = missing_market_dates(ticker, days=args.days) if args.days else missing_market_dates(ticker)
        if missing:
            print(f"{ticker}: missing {len(missing)} trading days: {missing[0]} .. {missing[-1]}")
        if not args.force_backfill and not needs_postgres_catchup(ticker, lookback_days=args.days):
            print(f"{ticker}: already current (market_date={postgres_latest_market_date(ticker)})")
            continue

        report = sync_postgres_snapshots(ticker, force_backfill=args.force_backfill)
        print(json.dumps(report, indent=2, default=str))
        after = count_snapshots(ticker)
        latest_after = latest_timestamp(ticker)
        print(f"{ticker}: now {after} snapshots (+{after - before}), latest={latest_after or 'none'}")
        if report.get("reason") and report.get("reason") not in {"up_to_date", None}:
            if report.get("backfill_started") and "failed" in str(report.get("reason", "")).lower():
                ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
