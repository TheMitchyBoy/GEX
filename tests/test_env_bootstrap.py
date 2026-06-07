import importlib

import pytest


def test_parse_env_minutes_accepts_leading_decimal(monkeypatch):
    monkeypatch.setenv("GEX_REFRESH_INTERVAL_MINUTES", ".5")
    from gex_core.env_bootstrap import parse_env_minutes

    assert parse_env_minutes("GEX_REFRESH_INTERVAL_MINUTES", 10.0) == 0.5


def test_parse_env_minutes_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("GEX_REFRESH_INTERVAL_MINUTES", "not-a-number")
    from gex_core.env_bootstrap import parse_env_minutes

    assert parse_env_minutes("GEX_REFRESH_INTERVAL_MINUTES", 10.0) == 10.0


def test_refresh_module_imports_with_fractional_interval(monkeypatch):
    monkeypatch.setenv("GEX_REFRESH_INTERVAL_MINUTES", ".5")
    import gex_core.refresh as refresh

    importlib.reload(refresh)
    assert refresh.DEFAULT_REFRESH_MINUTES == 0.5
