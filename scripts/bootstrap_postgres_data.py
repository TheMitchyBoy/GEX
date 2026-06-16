#!/usr/bin/env python3
"""Bootstrap Postgres snapshot history from CSV exports and UW API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()

from gex_core.bootstrap_data import bootstrap_postgres_data


def main() -> int:
    report = bootstrap_postgres_data()
    print(json.dumps(report, indent=2, default=str))
    if not report.get("postgres"):
        return 1
    if report.get("reason") == "DATABASE_URL not set":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
