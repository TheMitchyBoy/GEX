#!/usr/bin/env python3
"""Refresh GEX CSV exports for configured tickers (no SQLite)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.refresh import DEFAULT_TICKERS, refresh_tickers


def main():
    parser = argparse.ArgumentParser(description="Refresh GEX CSV exports")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help="Comma-separated tickers")
    parser.add_argument("--force", action="store_true", help="Ignore staleness and refresh")
    parser.add_argument("--cboe", action="store_true", help="Force CBOE (full surface) instead of UW")
    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    results = refresh_tickers(tickers, force=args.force, force_cboe=args.cboe)
    for ticker, ok in results.items():
        print(f"{ticker}: {'ok' if ok else 'failed'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
