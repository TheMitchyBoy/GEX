from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from gex_core.market_time import bars_held_since_entry, is_trader_session_active
from gex_core.trading.config import trader_bar_minutes, trader_cycle_seconds


def test_trader_cycle_seconds_default(monkeypatch):
    monkeypatch.delenv("GEX_TRADER_CYCLE_SECONDS", raising=False)
    assert trader_cycle_seconds() == 30


def test_trader_cycle_seconds_zero_disables_fast_loop(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_CYCLE_SECONDS", "0")
    assert trader_cycle_seconds() == 0


def test_trader_bar_minutes_follows_refresh(monkeypatch):
    monkeypatch.setenv("GEX_REFRESH_INTERVAL_MINUTES", "10")
    monkeypatch.delenv("GEX_TRADER_BAR_MINUTES", raising=False)
    assert trader_bar_minutes() == 10.0


def test_bars_held_since_entry():
    entry = datetime.now().replace(tzinfo=ZoneInfo("UTC")).isoformat()
    assert bars_held_since_entry(entry, bar_minutes=10) == 0


def test_is_trader_session_active_weekday_rth(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_SESSION_ONLY", "1")
    dt = datetime(2026, 6, 5, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    assert is_trader_session_active(now=dt)


def test_is_trader_session_active_closed_after_hours(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_SESSION_ONLY", "1")
    dt = datetime(2026, 6, 5, 17, 0, tzinfo=ZoneInfo("America/New_York"))
    assert not is_trader_session_active(now=dt)


def test_is_trader_session_active_can_disable(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_SESSION_ONLY", "0")
    dt = datetime(2026, 6, 5, 20, 0, tzinfo=ZoneInfo("America/New_York"))
    assert is_trader_session_active(now=dt)
