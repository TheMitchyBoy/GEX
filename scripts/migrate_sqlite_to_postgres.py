#!/usr/bin/env python3
"""Migrate SQLite journals and export index metadata into Railway PostgreSQL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()

from gex_core.data_root import configure_data_paths
from gex_core.db import use_postgres
from gex_core.migrate_postgres import migrate_sqlite_to_postgres, should_auto_migrate, sqlite_has_data


def _print_stats(stats) -> None:
    if stats.skipped:
        print(f"Migration skipped: {stats.reason}")
        return
    print("Migration complete:")
    print(f"  snapshots:       {stats.snapshots}")
    print(f"  trades:          {stats.trades}")
    print(f"  decisions:       {stats.decisions}")
    print(f"  trader_state:    {stats.trader_state}")
    print(f"  llm_predictions: {stats.llm_predictions}")
    print(f"  daily_insights:  {stats.daily_insights}")
    print(f"  export index:    +{stats.export_sync_added} from disk")
    print(f"  total migrated:  {stats.total_rows()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Migrate even when PostgreSQL already has rows",
    )
    parser.add_argument(
        "--no-export-sync",
        action="store_true",
        help="Skip reconciling CSV/JSON exports into the Postgres index",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print whether auto-migration would run and exit",
    )
    parser.add_argument(
        "--if-needed",
        action="store_true",
        help="Exit without work unless auto-migration is recommended",
    )
    args = parser.parse_args()

    configure_data_paths()

    if not use_postgres():
        print("DATABASE_URL is not set — cannot migrate to PostgreSQL.")
        return 1

    if args.check:
        print(f"sqlite_has_data={sqlite_has_data()}")
        print(f"should_auto_migrate={should_auto_migrate()}")
        return 0

    if args.if_needed and not should_auto_migrate():
        print("Migration not needed (set GEX_MIGRATE_SQLITE=1 to force).")
        return 0

    stats = migrate_sqlite_to_postgres(force=args.force, sync_exports=not args.no_export_sync)
    _print_stats(stats)
    return 0 if not stats.skipped or stats.reason == "No SQLite data to migrate" else 0


if __name__ == "__main__":
    raise SystemExit(main())
