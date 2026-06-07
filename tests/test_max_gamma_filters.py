"""Tests for max-gamma direction filters and magnet-primary exits."""

from gex_core.trading.exits import ExitProfile, ExitState, evaluate_exit
from gex_core.trading.filters import MarketContext, evaluate_entry_filters


def test_momentum_filter_blocks_call_when_spot_falling(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "1")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_FLOW_ALIGN", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_MOMENTUM", "1")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_FLIP_SIDE", "0")
    monkeypatch.setenv("GEX_TRADER_MOMENTUM_BARS", "2")
    signals = {
        "available": True,
        "spot": 5000.0,
        "master_direction": "call",
        "recommended": {
            "strike": 5010.0,
            "option_type": "call",
            "gamma_delta": 0.1,
            "score": 1.0,
        },
    }
    ctx = MarketContext(spot=5000.0, spot_history=(5010.0, 5005.0, 5000.0))
    result = evaluate_entry_filters(signals, market=ctx)
    assert not result["approve"]
    assert result["filter"] == "momentum"


def test_flip_filter_blocks_call_below_gamma_flip(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "1")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_FLOW_ALIGN", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_MOMENTUM", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_FLIP_SIDE", "1")
    signals = {
        "available": True,
        "spot": 4990.0,
        "master_direction": "call",
        "recommended": {
            "strike": 5010.0,
            "option_type": "call",
            "gamma_delta": 0.1,
            "score": 1.0,
        },
    }
    ctx = MarketContext(spot=4990.0, gamma_flip=5000.0, spot_history=(4985.0, 4990.0))
    result = evaluate_entry_filters(signals, market=ctx)
    assert not result["approve"]
    assert result["filter"] == "gamma_flip"


def test_magnet_primary_exit_at_breakeven(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MAX_GAMMA_ONLY", "1")
    monkeypatch.setenv("GEX_TRADER_MAGNET_TOUCH_EXIT", "1")
    profile = ExitProfile(hold_for_target=True, full_take_profit=0.60)
    state = ExitState()
    reason, _ = evaluate_exit(
        0.02,
        state=state,
        bars_held=3,
        entry_spot=5000.0,
        strike=5010.0,
        current_spot=5050.0,
        option_type="call",
        profile=profile,
        magnet_strike=5050.0,
    )
    assert reason is None

    reason, pnl = evaluate_exit(
        0.10,
        state=state,
        bars_held=3,
        entry_spot=5000.0,
        strike=5010.0,
        current_spot=5050.0,
        option_type="call",
        profile=profile,
        magnet_strike=5050.0,
    )
    assert reason == "magnet_touch"
    assert pnl == 0.10
