#!/usr/bin/env python3
"""Create GEX tables in Railway PostgreSQL (idempotent)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()

from gex_core.db import database_url, ensure_postgres_schema, use_postgres


def main() -> int:
    if not use_postgres():
        print("DATABASE_URL is not set — skipping PostgreSQL schema init (SQLite mode).")
        return 0
    ensure_postgres_schema()
    print(f"PostgreSQL schema ready ({database_url()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
