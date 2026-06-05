#!/usr/bin/env python3
"""Refresh GEX CSV exports from Unusual Whales."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.env_bootstrap import bootstrap_env
from gex_core.intraday_backfill import backfill_recent_intraday
from gex_core.refresh import DEFAULT_TICKERS, refresh_recent_tickers, refresh_tickers

bootstrap_env()


def main():
    parser = argparse.ArgumentParser(description="Fetch UW GEX and save CSV exports")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help="Comma-separated tickers")
    parser.add_argument("--force", action="store_true", help="Ignore staleness and refresh")
    parser.add_argument("--market-date", help="Fetch a historical UW market date in YYYY-MM-DD format")
    parser.add_argument("--backfill-days", type=int, help="Fetch one EOD snapshot per recent calendar date")
    parser.add_argument(
        "--intraday-days",
        type=int,
        help="Backfill UW spot-exposures for recent weekdays (advanced API tier)",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=int(os.environ.get("GEX_BACKFILL_INTERVAL_MINUTES", "10")),
        help="Sample UW rows every N minutes during intraday backfill",
    )
    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.intraday_days:
        total = 0
        for ticker in tickers:
            results_by_date = backfill_recent_intraday(
                ticker,
                days=args.intraday_days,
                force=args.force,
                interval_minutes=args.interval_minutes,
            )
            day_total = sum(results_by_date.values())
            total += day_total
            print(
                f"{ticker} intraday: saved {day_total} "
                f"{args.interval_minutes}-min snapshots"
            )
        print(f"intraday backfill total: {total} snapshots")
        sys.exit(0 if total > 0 else 1)

    if args.backfill_days:
        results_by_date = refresh_recent_tickers(tickers, days=args.backfill_days, force=args.force)
        ok_values = []
        for market_date, results in results_by_date.items():
            for ticker, ok in results.items():
                ok_values.append(ok)
                print(f"{market_date} {ticker}: {'ok' if ok else 'skipped/failed'}")
        sys.exit(0 if any(ok_values) else 1)

    results = refresh_tickers(tickers, force=args.force, market_date=args.market_date)
    for ticker, ok in results.items():
        print(f"{ticker}: {'ok' if ok else 'skipped/failed'}")
    sys.exit(0 if any(results.values()) else 1)


if __name__ == "__main__":
    main()
