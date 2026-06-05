#!/usr/bin/env python3
"""Compact old GEX exports: drop bulky strike CSVs, keep summaries + cumulative."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.exports import EXPORT_DIR, TIMESTAMP_RE, parse_timestamp
from gex_core.storage import sync_ticker_exports


def compact_exports(
    ticker: str,
    *,
    keep_full_days: int = 14,
    export_dir: Path | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    export_dir = export_dir or EXPORT_DIR
    ticker = ticker.upper()
    cutoff = datetime.now() - timedelta(days=keep_full_days)
    removed = 0
    kept = 0

    for path in sorted(export_dir.glob(f"{ticker}_gex_by_strike_*.csv")):
        match = TIMESTAMP_RE.match(path.name)
        if not match:
            continue
        ts = match.group("ts")
        if parse_timestamp(ts) >= cutoff:
            kept += 1
            continue
        if dry_run:
            print(f"would remove {path.name}")
            removed += 1
            continue
        path.unlink(missing_ok=True)
        surface = export_dir / f"{ticker}_gex_surface_{ts}.csv"
        surface.unlink(missing_ok=True=True)
        removed += 1

    if not dry_run:
        sync_ticker_exports(ticker, export_dir)
    return removed, kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact aged GEX strike exports.")
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--keep-full-days", type=int, default=14)
    parser.add_argument("--export-dir", default=str(EXPORT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    removed, kept = compact_exports(
        args.ticker,
        keep_full_days=args.keep_full_days,
        export_dir=Path(args.export_dir),
        dry_run=args.dry_run,
    )
    print(f"Removed {removed} aged strike files; kept {kept} within window.")


if __name__ == "__main__":
    main()
