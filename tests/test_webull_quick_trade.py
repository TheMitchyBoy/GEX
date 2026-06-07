"""Tests for Webull quick-trade pricing and conditions."""

from gex_core.trading.webull_quick_trade import (
    analyze_quote,
    entry_conditions,
    entry_limit_price,
    exit_conditions,
    exit_limit_price,
    price_ladder,
)


def test_analyze_quote_mid_and_spread():
    q = analyze_quote({"bid": 1.0, "ask": 1.2, "last": 1.1, "symbol": "SPY260606C00590000"})
    assert q.mid == 1.1
    assert q.spread == 0.2
    assert abs(q.spread_pct - 0.2 / 1.1) < 0.001


def test_entry_smart_prefers_mid_on_tight_spread():
    analysis = analyze_quote({"bid": 2.0, "ask": 2.1, "symbol": "X"})
    px = entry_limit_price(analysis, style="smart")
    assert px == 2.05


def test_exit_smart_prefers_mid_on_tight_spread():
    analysis = analyze_quote({"bid": 2.0, "ask": 2.1, "symbol": "X"})
    px = exit_limit_price(analysis, style="smart", entry_premium=1.5)
    assert px == 2.05


def test_entry_conditions_tight_spread_is_go():
    analysis = analyze_quote({"bid": 1.0, "ask": 1.05, "symbol": "X"})
    cond = entry_conditions(analysis)
    assert cond["action"] in {"go", "wait"}
    assert cond["score"] >= 0.65


def test_exit_conditions_take_profit(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_TAKE_PROFIT_PCT", "0.12")
    analysis = analyze_quote({"bid": 1.4, "ask": 1.5, "symbol": "X"})
    cond = exit_conditions(analysis, entry_premium=1.0)
    assert cond["action"] == "sell"
    assert cond["pnl_pct"] >= 0.12


def test_price_ladder_has_four_styles():
    analysis = analyze_quote({"bid": 1.0, "ask": 1.2, "symbol": "X"})
    ladder = price_ladder(analysis, entry_premium=1.0)
    assert set(ladder["entry"].keys()) == {"passive", "mid", "smart", "aggressive"}
    assert set(ladder["exit"].keys()) == {"passive", "mid", "smart", "aggressive"}
