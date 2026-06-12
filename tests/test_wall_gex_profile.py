"""Wall GEX profile resolution for full vs near-spot windows."""

from gex_core.trading.config import (
    DEFAULT_WALL_WINDOW_PCT,
    is_near_wall_window,
    near_wall_window_pct,
    wall_gex_profile,
)


def test_is_near_wall_window():
    assert is_near_wall_window(near_wall_window_pct())
    assert not is_near_wall_window(DEFAULT_WALL_WINDOW_PCT)


def test_wall_gex_profile_near_defaults(monkeypatch):
    monkeypatch.delenv("GEX_NEAR_WALL_TAKE_PROFIT_PCT", raising=False)
    monkeypatch.delenv("GEX_NEAR_WALL_MAX_HOLD_BARS", raising=False)
    monkeypatch.delenv("GEX_NEAR_WALL_REENTER_ON_SHIFT", raising=False)
    profile = wall_gex_profile(0.01)
    assert profile.near is True
    assert profile.take_profit_pct == 0.28
    assert profile.max_hold_bars == 10
    assert profile.reenter_on_shift is False


def test_wall_gex_profile_full_defaults(monkeypatch):
    monkeypatch.delenv("GEX_WALL_TAKE_PROFIT_PCT", raising=False)
    monkeypatch.delenv("GEX_WALL_MAX_HOLD_BARS", raising=False)
    profile = wall_gex_profile(DEFAULT_WALL_WINDOW_PCT)
    assert profile.near is False
    assert profile.take_profit_pct == 0.22
    assert profile.max_hold_bars == 8
