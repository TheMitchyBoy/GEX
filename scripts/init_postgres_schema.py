#!/usr/bin/env python3
"""Create GEX tables in Railway PostgreSQL (idempotent)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()

from gex_core.db import POSTGRES_TABLES, PROCESSOR_POSTGRES_TABLES, ensure_postgres_schema, use_postgres
from gex_core.runtime_mode import is_processor_mode


def main() -> int:
    if not use_postgres():
        print("DATABASE_URL is not set — skipping PostgreSQL schema init (SQLite mode).")
        print("To create tables manually, paste scripts/postgres_schema.sql into Railway Postgres.")
        return 0

    processor_only = is_processor_mode()
    tables = ensure_postgres_schema(processor_only=processor_only)
    expected = PROCESSOR_POSTGRES_TABLES if processor_only else POSTGRES_TABLES
    missing = [name for name in expected if name not in tables]
    print("PostgreSQL schema ready.")
    print("Tables:", ", ".join(tables) if tables else "(none)")
    if missing:
        print("WARNING: missing expected tables:", ", ".join(missing))
        return 1
    print("All expected tables present:", ", ".join(expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
