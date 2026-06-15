#!/usr/bin/env python3
"""Create GEX tables in Railway PostgreSQL (idempotent)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()

from gex_core.db import POSTGRES_TABLES, ensure_postgres_schema, use_postgres


def main() -> int:
    if not use_postgres():
        print("DATABASE_URL is not set — skipping PostgreSQL schema init (SQLite mode).")
        print("To create tables manually, paste scripts/postgres_schema.sql into Railway Postgres.")
        return 0

    tables = ensure_postgres_schema()
    missing = [name for name in POSTGRES_TABLES if name not in tables]
    print("PostgreSQL schema ready.")
    print("Tables:", ", ".join(tables) if tables else "(none)")
    if missing:
        print("WARNING: missing expected tables:", ", ".join(missing))
        return 1
    print("All expected tables present:", ", ".join(POSTGRES_TABLES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
