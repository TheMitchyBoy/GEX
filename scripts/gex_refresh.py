#!/usr/bin/env python3
"""Refresh GEX CSV exports from Unusual Whales."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.refresh import DEFAULT_TICKERS, refresh_tickers


def main():
    parser = argparse.ArgumentParser(description="Fetch UW GEX and save CSV exports")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help="Comma-separated tickers")
    parser.add_argument("--force", action="store_true", help="Ignore staleness and refresh")
    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    results = refresh_tickers(tickers, force=args.force)
    for ticker, ok in results.items():
        print(f"{ticker}: {'ok' if ok else 'failed'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
