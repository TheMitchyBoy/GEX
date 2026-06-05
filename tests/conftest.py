"""Shared pytest configuration."""

from __future__ import annotations

import os


def pytest_configure() -> None:
    # Keep dashboard/history tests fast when minute-level backfills exist locally.
    os.environ.setdefault("GEX_HISTORY_LOOKBACK_DAYS", "1")
    os.environ.setdefault("GEX_HISTORY_MAX_SNAPSHOTS", "60")
