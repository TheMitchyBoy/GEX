"""Tests for Webull quick-trade pricing and conditions."""

from gex_core.trading.webull_quick_trade import (
    _maybe_map_strike_for_execution,
    analyze_quote,
    build_recommended_trade,
    combined_entry_guidance,
    entry_conditions,
    entry_limit_price,
    estimate_option_quote,
    exit_conditions,
    exit_limit_price,
    price_ladder,
    quote_payload,
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


def test_build_recommended_trade_unavailable_without_signal():
    out = build_recommended_trade(strategy_state={"signals": {"available": False, "reason": "no data"}})
    assert out["available"] is False
    assert "no data" in out["reason"]


def test_build_recommended_trade_maps_execution_strike(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_EXECUTION_TICKER", "SPY")
    monkeypatch.setenv("GEX_TRADER_SIGNAL_TICKER", "SPX")
    state = {
        "spot": 5000.0,
        "signals": {
            "available": True,
            "recommended": {
                "strike": 5000.0,
                "option_type": "call",
                "signal_type": "max_positive_gamma",
                "rationale": "magnet",
                "gamma_delta": 0.1,
                "magnet_strike": 5000.0,
            },
        },
        "filters": {"approve": True, "reason": "ok"},
        "advice": {"approve": True, "reason": "go"},
    }
    out = build_recommended_trade(strategy_state=state)
    assert out["available"] is True
    assert out["signal_strike"] == 5000.0
    assert out["execution_strike"] > 0
    assert out["strategy_ready"] is True
    assert out["symbol"].startswith("SPY")


def test_estimate_option_quote_builds_two_sided_nbbo():
    q = estimate_option_quote(symbol="SPY260608C00758000", spot=758.0, strike=758.0)
    assert q["bid"] and q["ask"] and q["last"]
    assert q["bid"] < q["last"] < q["ask"]


def test_maybe_map_strike_for_execution_maps_spx_to_spy(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_EXECUTION_TICKER", "SPY")
    monkeypatch.setenv("GEX_TRADER_SIGNAL_TICKER", "SPX")
    strategy = {
        "signal_strike": 7600.0,
        "execution_strike": 758.0,
        "signal_spot": 7580.0,
        "execution_spot": 757.0,
    }
    mapped = _maybe_map_strike_for_execution("SPY", 7600.0, strategy_trade=strategy)
    assert mapped == 758.0


def test_quote_payload_uses_estimate_when_live_missing(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_EXECUTION_TICKER", "SPY")
    monkeypatch.setenv("GEX_TRADER_SIGNAL_TICKER", "SPX")
    monkeypatch.setattr("gex_core.trading.webull_quick_trade.webull_configured", lambda: False)
    strategy = {
        "available": True,
        "signal_strike": 7600.0,
        "execution_strike": 758.0,
        "signal_spot": 7580.0,
        "execution_spot": 757.0,
        "strategy_ready": True,
        "filters": {"approve": True},
        "advice": {"approve": True, "reason": "aligned"},
    }
    payload = quote_payload(
        underlying="SPY",
        option_type="call",
        strike=7600.0,
        expire_date="2026-06-08",
        strategy_trade=strategy,
    )
    assert payload["strike"] == 758.0
    assert payload["quote_source"] == "estimated"
    assert payload["quote"]["mid"] is not None
    assert payload["entry_conditions"]["signal"] != "no_quote"


def test_combined_entry_guidance_boosts_when_strategy_ready():
    quote_entry = {"action": "wait", "score": 0.55, "summary": "Moderate spread"}
    strategy = {
        "available": True,
        "strategy_ready": True,
        "rationale": "Max gamma magnet",
        "signal_type": "max_positive_gamma",
        "filters": {"approve": True},
        "advice": {"approve": True, "reason": "aligned"},
    }
    merged = combined_entry_guidance(quote_entry, strategy_trade=strategy)
    assert merged["strategy"]["ready"] is True
    assert merged["score"] >= 0.7
