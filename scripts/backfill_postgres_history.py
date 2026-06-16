#!/usr/bin/env python3
"""Backfill UW API GEX history directly into PostgreSQL snapshot tables."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()

from gex_core.data_root import configure_data_paths
from gex_core.db import ensure_postgres_schema, use_postgres
from gex_core.intraday_backfill import backfill_recent_daily, backfill_recent_intraday
from gex_core.refresh import DEFAULT_TICKERS
from gex_core.storage import count_snapshots


def _parse_tickers(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_TICKERS)
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def _configure_backfill_env() -> None:
    os.environ.setdefault("GEX_PROCESSOR_MODE", "1")
    os.environ.setdefault("GEX_EXPORT_CSV", "0")
    os.environ.setdefault("GEX_SKIP_DUPLICATE_SNAPSHOTS", "1")
    os.environ.setdefault("GEX_BACKFILL_MODE", "1")
    # Scaled intraday profiles can disagree slightly with minute totals.
    os.environ.setdefault("GEX_HARD_REJECT_TOTAL_GEX_MISMATCH", "0")
    os.environ.setdefault("GEX_MIN_STRIKE_COUNT", "3")


def _sparse_threshold() -> int:
    try:
        return int(os.environ.get("GEX_BACKFILL_MIN_SNAPSHOTS", "30"))
    except (TypeError, ValueError):
        return 30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--intraday-days", type=int, default=int(os.environ.get("GEX_INTRADAY_BACKFILL_DAYS", "90")))
    parser.add_argument("--daily-days", type=int, default=int(os.environ.get("GEX_DAILY_BACKFILL_DAYS", "90")))
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=int(os.environ.get("GEX_BACKFILL_INTERVAL_MINUTES", "10")),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing timestamps")
    parser.add_argument(
        "--if-sparse",
        action="store_true",
        help="Skip tickers that already have GEX_BACKFILL_MIN_SNAPSHOTS rows in Postgres",
    )
    parser.add_argument("--intraday-only", action="store_true")
    parser.add_argument("--daily-only", action="store_true")
    args = parser.parse_args()

    configure_data_paths()
    _configure_backfill_env()

    if not use_postgres():
        print("DATABASE_URL is not set — cannot backfill into PostgreSQL.")
        return 1

    ensure_postgres_schema()
    tickers = _parse_tickers(args.tickers)
    min_snapshots = _sparse_threshold()
    ok = False

    for ticker in tickers:
        count = count_snapshots(ticker)
        if args.if_sparse and not args.force and count >= min_snapshots:
            print(f"{ticker}: skipped (already has {count} snapshots, need {min_snapshots})")
            continue

        since_date = "" if args.if_sparse and not args.force else None

        if not args.daily_only and args.intraday_days > 0:
            results = backfill_recent_intraday(
                ticker,
                days=args.intraday_days,
                force=args.force,
                interval_minutes=args.interval_minutes,
                since_date=since_date,
            )
            total = sum(results.values())
            print(
                f"{ticker} intraday: saved {total} "
                f"{args.interval_minutes}-min snapshots across {len(results)} days"
            )
            for day, count in results.items():
                if count:
                    print(f"  {day}: {count}")
            ok = ok or total > 0

        if not args.intraday_only and args.daily_days > 0:
            results = backfill_recent_daily(ticker, days=args.daily_days, force=args.force)
            saved = sum(1 for value in results.values() if value)
            print(f"{ticker} daily: saved {saved}/{len(results)} EOD strike snapshots")
            ok = ok or saved > 0

        print(f"{ticker}: postgres snapshot count now {count_snapshots(ticker)}")

    return 0 if ok or args.if_sparse else 1


if __name__ == "__main__":
    raise SystemExit(main())
