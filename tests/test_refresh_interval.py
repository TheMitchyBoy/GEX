import importlib

import pytest

import gex_core.refresh as refresh_mod


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10", 10.0),
        (".5", 0.5),
        ("0.5", 0.5),
        ("  1.25  ", 1.25),
    ],
)
def test_refresh_interval_minutes_parses_fractions(monkeypatch, raw, expected):
    monkeypatch.setenv("GEX_REFRESH_INTERVAL_MINUTES", raw)
    assert refresh_mod.refresh_interval_minutes() == pytest.approx(expected)


def test_refresh_interval_minutes_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("GEX_REFRESH_INTERVAL_MINUTES", "not-a-number")
    assert refresh_mod.refresh_interval_minutes(default="10") == pytest.approx(10.0)


def test_default_refresh_minutes_reads_env_at_import(monkeypatch):
    monkeypatch.setenv("GEX_REFRESH_INTERVAL_MINUTES", ".5")
    reloaded = importlib.reload(refresh_mod)
    assert reloaded.DEFAULT_REFRESH_MINUTES == pytest.approx(0.5)
    monkeypatch.delenv("GEX_REFRESH_INTERVAL_MINUTES")
    importlib.reload(refresh_mod)
