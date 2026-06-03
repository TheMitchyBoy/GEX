#!/usr/bin/env python3
"""Refresh configured tickers and persist GEX snapshots to the database."""

from __future__ import annotations

import argparse

from gex_db.refresh import DEFAULT_TICKERS, refresh_tickers
from gex_db.store import import_csv_exports, init_db


def main():
    parser = argparse.ArgumentParser(description="Refresh GEX snapshots in the database.")
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated tickers (default: env GEX_DEFAULT_TICKERS or SPX)",
    )
    parser.add_argument("--force", action="store_true", help="Refresh even if the latest snapshot is fresh")
    parser.add_argument(
        "--import-csv",
        action="store_true",
        help="Import existing CSV exports before refreshing",
    )
    args = parser.parse_args()

    init_db()
    if args.import_csv:
        imported = import_csv_exports()
        print(f"Imported {imported} CSV snapshots")

    tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()]
    results = refresh_tickers(tickers, force=args.force)
    for ticker, ok in results.items():
        print(f"{ticker}: {'updated' if ok else 'skipped or failed'}")


if __name__ == "__main__":
    main()
