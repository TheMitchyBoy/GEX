#!/usr/bin/env python3
"""Refresh GEX CSV exports from Unusual Whales."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.refresh import DEFAULT_TICKERS, refresh_recent_tickers, refresh_tickers


def main():
    parser = argparse.ArgumentParser(description="Fetch UW GEX and save CSV exports")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help="Comma-separated tickers")
    parser.add_argument("--force", action="store_true", help="Ignore staleness and refresh")
    parser.add_argument("--market-date", help="Fetch a historical UW market date in YYYY-MM-DD format")
    parser.add_argument("--backfill-days", type=int, help="Fetch one snapshot per recent calendar date")
    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
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
