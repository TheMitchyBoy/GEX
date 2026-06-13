"""Shared pytest configuration."""

from __future__ import annotations

import os

import pytest


def pytest_configure() -> None:
    # Keep dashboard/history tests fast when minute-level backfills exist locally.
    os.environ.setdefault("GEX_HISTORY_LOOKBACK_DAYS", "1")
    os.environ.setdefault("GEX_HISTORY_MAX_SNAPSHOTS", "60")
    os.environ.setdefault("GEX_DISABLE_SCHEDULER", "1")
    os.environ.setdefault("GEX_WEBULL_ENABLED", "0")


@pytest.fixture(autouse=True)
def _enable_webull_in_webull_tests(monkeypatch, request):
    if "test_webull" in request.node.nodeid:
        monkeypatch.setenv("GEX_WEBULL_ENABLED", "1")
