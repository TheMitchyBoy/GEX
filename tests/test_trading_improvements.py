"""Tests for trading strategy improvements."""

from __future__ import annotations

import pandas as pd
import pytest

from gex_core.market_time import export_ts_entry_window_ok, is_eod_flatten_time
from gex_core.trading.config import momentum_bars
from gex_core.trading.exits import ExitProfile, ExitState, evaluate_exit, resolve_full_take_profit
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


def test_zero_dte_filter_blocks_low_ratio(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MIN_ZERO_DTE_RATIO", "0.4")
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
        prev_spot=4995.0,
        zero_dte_ratio=0.2,
        spot_history=(4995.0, 5000.0),
        export_ts="2026-06-05_143000",
    )
    result = evaluate_entry_filters(signals, market=ctx)
    assert not result["approve"]
    assert result["filter"] == "zero_dte"


def test_iv_rank_filter_blocks_high_iv(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MAX_IV_RANK", "0.85")
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
        prev_spot=4995.0,
        iv_rank=0.92,
        spot_history=(4995.0, 5000.0),
        export_ts="2026-06-05_143000",
    )
    result = evaluate_entry_filters(signals, market=ctx)
    assert not result["approve"]
    assert result["filter"] == "iv_rank"


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


def test_multi_bar_momentum_requires_trend(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MOMENTUM_BARS", "2")
    _ = momentum_bars()
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
    flat_ctx = MarketContext(
        spot=5000.0,
        spot_history=(5002.0, 5001.0, 5000.0),
        regime="LONG gamma",
        export_ts="2026-06-05_143000",
    )
    assert not evaluate_entry_filters(signals, market=flat_ctx)["approve"]

    rising_ctx = MarketContext(
        spot=5000.0,
        spot_history=(4996.0, 4998.0, 5000.0),
        regime="LONG gamma",
        export_ts="2026-06-05_143000",
    )
    assert evaluate_entry_filters(signals, market=rising_ctx)["approve"]


def test_entry_window_midday_ok():
    assert export_ts_entry_window_ok("2026-06-05_143000")


def test_event_day_size_multiplier(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_BLOCK_EVENTS", "1")
    monkeypatch.setenv("GEX_TRADER_EVENT_SIZE_MULT", "0.5")
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
        prev_spot=4995.0,
        is_nfp_day=True,
        spot_history=(4995.0, 5000.0),
        export_ts="2026-06-05_143000",
    )
    result = evaluate_entry_filters(signals, market=ctx)
    assert result["approve"]
    assert result["size_multiplier"] == 0.5
