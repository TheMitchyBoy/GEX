"""Tests for trading strategy improvements."""

from __future__ import annotations

import pandas as pd
import pytest

from gex_core.market_time import export_ts_entry_window_ok, is_eod_flatten_time
from gex_core.trading.exits import ExitProfile, ExitState, evaluate_exit, resolve_full_take_profit
from gex_core.trading.config import require_flow_alignment
from gex_core.trading.filters import MarketContext, evaluate_entry_filters
from gex_core.trading.sizing import resolve_contract_qty


@pytest.fixture(autouse=True)
def enable_strict_filters(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "1")
    monkeypatch.setenv("GEX_TRADER_ENTRY_TIME_FILTER", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_ZERO_DTE_RATIO", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_FLIP_SIDE", "0")
    monkeypatch.setenv("GEX_TRADER_MAX_IV_RANK", "1")
    monkeypatch.setenv("GEX_TRADER_MIN_MAGNET_PROGRESS", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    monkeypatch.setenv("GEX_TRADER_PREFER_SIGNAL", "")
    monkeypatch.setenv("GEX_TRADER_MIN_FLOW_AGGRESSIVENESS", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_FLOW_BUY_RATIO", "0")


def test_require_flow_alignment_default_off(monkeypatch):
    monkeypatch.delenv("GEX_TRADER_REQUIRE_FLOW_ALIGN", raising=False)
    assert not require_flow_alignment()


def test_risk_based_sizing_caps_contracts(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_RISK_SIZING", "1")
    monkeypatch.setenv("GEX_TRADER_ACCOUNT_EQUITY", "500")
    monkeypatch.setenv("GEX_TRADER_RISK_PER_TRADE_PCT", "0.02")
    qty = resolve_contract_qty(
        confidence=0.9,
        premium=1.35,
        entry_spot=737.0,
        strike=738.0,
        account_equity=500.0,
    )
    assert qty >= 1
    assert qty <= 2


def test_gamma_delta_filter_blocks_weak_signal(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0.05")
    signals = {
        "available": True,
        "spot": 5000.0,
        "recommended": {
            "strike": 5010.0,
            "option_type": "call",
            "gamma_delta": 0.02,
            "score": 1.0,
            "signal_type": "max_positive_gamma",
        },
    }
    ctx = MarketContext(spot=5000.0, prev_spot=4995.0, spot_history=(4995.0, 5000.0))
    result = evaluate_entry_filters(signals, market=ctx)
    assert not result["approve"]
    assert result["filter"] == "gamma_delta"


def test_flow_filter_blocks_misaligned_call(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_REQUIRE_FLOW_ALIGN", "1")
    monkeypatch.setenv("GEX_TRADER_MIN_FLOW_BUY_RATIO", "0.55")
    signals = {
        "available": True,
        "spot": 5000.0,
        "recommended": {
            "strike": 5010.0,
            "option_type": "call",
            "gamma_delta": 0.2,
            "score": 1.0,
            "signal_type": "max_positive_gamma",
        },
    }
    ctx = MarketContext(
        spot=5000.0,
        flow_net_delta_gex_bn=-0.5,
        flow_buy_ratio=0.40,
        spot_history=(4995.0, 5000.0),
    )
    result = evaluate_entry_filters(signals, market=ctx)
    assert not result["approve"]
    assert result["filter"] == "flow"


def test_flow_and_gamma_delta_pass(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0.03")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_FLOW_ALIGN", "1")
    signals = {
        "available": True,
        "spot": 5000.0,
        "recommended": {
            "strike": 5010.0,
            "option_type": "call",
            "gamma_delta": 0.2,
            "score": 1.0,
            "signal_type": "max_positive_gamma",
        },
    }
    ctx = MarketContext(
        spot=5000.0,
        flow_net_delta_gex_bn=0.5,
        flow_buy_ratio=0.60,
        spot_history=(4995.0, 5000.0),
    )
    result = evaluate_entry_filters(signals, market=ctx)
    assert result["approve"]


def test_magnet_touch_exit_triggers():
    profile = ExitProfile(full_take_profit=0.35)
    state = ExitState()
    reason, pnl = evaluate_exit(
        0.08,
        state=state,
        bars_held=2,
        entry_spot=5000.0,
        strike=5010.0,
        current_spot=5010.0,
        option_type="call",
        profile=profile,
    )
    assert reason == "magnet_touch"


def test_dynamic_take_profit_scales_down():
    profile = ExitProfile(full_take_profit=0.35)
    tp = resolve_full_take_profit(profile, expected_move_pct=0.005)
    assert tp < 0.35
    assert tp >= 0.08


def test_entry_window_midday_ok():
    assert export_ts_entry_window_ok("2026-06-05_143000")

