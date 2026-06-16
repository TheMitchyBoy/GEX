#!/usr/bin/env python3
"""Bulk-load on-disk GEX CSV/JSON exports into PostgreSQL snapshot tables."""

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
from gex_core.import_exports import import_ticker_exports, summarize_import_results
from gex_core.tickers import SUPPORTED_TICKERS


def _parse_tickers(raw: str | None) -> list[str]:
    if not raw:
        return list(SUPPORTED_TICKERS)
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", help="Comma-separated tickers (default: all supported)")
    parser.add_argument("--export-dir", type=Path, help="Override GEX_EXPORT_DIR")
    parser.add_argument("--force", action="store_true", help="Re-import even when ts already exists in Postgres")
    parser.add_argument("--dry-run", action="store_true", help="List work without writing")
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="Only import timestamps that are not already in Postgres (default behavior)",
    )
    args = parser.parse_args()

    configure_data_paths()

    if not use_postgres():
        print("DATABASE_URL is not set — cannot import into PostgreSQL.")
        return 1

    # Historical exports often have full-chain totals that differ after strike filtering.
    os.environ.setdefault("GEX_HARD_REJECT_TOTAL_GEX_MISMATCH", "0")
    os.environ.setdefault("GEX_SKIP_DUPLICATE_SNAPSHOTS", "0")
    os.environ.setdefault("GEX_MIN_STRIKE_COUNT", "3")

    ensure_postgres_schema()
    export_dir = args.export_dir
    tickers = _parse_tickers(args.tickers)
    totals = {"imported": 0, "skipped": 0, "errors": 0, "dry_run": 0}

    for ticker in tickers:
        results = import_ticker_exports(
            ticker,
            export_dir=export_dir,
            force=args.force,
            dry_run=args.dry_run,
            skip_existing=not args.force,
        )
        counts = summarize_import_results(results)
        for key, value in counts.items():
            totals[key] += value
        print(f"{ticker}: imported={counts['imported']} skipped={counts['skipped']} errors={counts['errors']}")
        for result in results:
            if result.status == "error":
                print(f"  error {result.ts}: {result.error}")

    print(
        "Done:"
        f" imported={totals['imported']}"
        f" skipped={totals['skipped']}"
        f" errors={totals['errors']}"
        f" dry_run={totals['dry_run']}"
    )
    return 1 if totals["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
