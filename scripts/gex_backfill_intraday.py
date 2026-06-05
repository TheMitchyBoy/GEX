#!/usr/bin/env python3
"""Backfill minute-level and daily UW GEX history for model training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.env_bootstrap import bootstrap_env
from gex_core.intraday_backfill import (
    backfill_recent_daily,
    backfill_recent_intraday,
    export_live_strike_snapshot,
)
from gex_core.refresh import DEFAULT_TICKERS

bootstrap_env()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill UW GEX history at minute or daily granularity")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--intraday-days", type=int, default=int(os.environ.get("GEX_INTRADAY_BACKFILL_DAYS", "90")))
    parser.add_argument("--daily-days", type=int, default=int(os.environ.get("GEX_DAILY_BACKFILL_DAYS", "90")))
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=int(os.environ.get("GEX_BACKFILL_INTERVAL_MINUTES", "10")),
        help="Sample UW 1-minute rows every N minutes (default 10)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing timestamps")
    parser.add_argument("--live-once", action="store_true", help="Fetch one live minute snapshot and exit")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    ok = False

    for ticker in tickers:
        if args.live_once:
            ts = export_live_strike_snapshot(ticker, force=args.force)
            print(f"{ticker} live: {'ok ' + ts if ts else 'failed'}")
            ok = ok or bool(ts)
            continue

        if args.intraday_days > 0:
            results = backfill_recent_intraday(
                ticker,
                days=args.intraday_days,
                force=args.force,
                interval_minutes=args.interval_minutes,
            )
            total = sum(results.values())
            print(
                f"{ticker} intraday: saved {total} "
                f"{args.interval_minutes}-min snapshots across {len(results)} days"
            )
            for day, count in results.items():
                print(f"  {day}: {count}")
            ok = ok or total > 0

        if args.daily_days > 0:
            results = backfill_recent_daily(ticker, days=args.daily_days, force=args.force)
            saved = sum(1 for v in results.values() if v)
            print(f"{ticker} daily: saved {saved}/{len(results)} EOD strike snapshots")
            ok = ok or saved > 0

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
