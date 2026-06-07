"""Tests for Monte Carlo trader config search."""

from __future__ import annotations

import pandas as pd

from scripts.monte_carlo_trader import TraderConfig, run_trial, score_result
from gex_core.trading.backtest import backtest_auto_trader


def _snapshot(ts: str, spot: float, strikes: dict[float, float]) -> dict:
    return {
        "ts": ts,
        "spot": spot,
        "strike": pd.Series(strikes),
        "regime": "LONG gamma",
        "gamma_flip": spot * 0.995,
        "flow_net_delta_gex_bn": 0.5,
        "interval_minutes": 10,
    }


def test_score_result_penalizes_zero_trades():
    assert score_result({"total_trades": 0}) < -1e8


def test_score_result_penalizes_session_gap_only():
    flat = score_result(
        {
            "total_trades": 3,
            "win_rate": 0.0,
            "total_pnl_usd": 0.0,
            "account": {"return_pct": 0.0, "max_drawdown_pct": 0.0},
            "by_exit_reason": {"session_gap": 3},
        }
    )
    assert flat < -1e5


def test_score_result_prefers_positive_return():
    low = score_result({"total_trades": 5, "win_rate": 0.4, "total_pnl_usd": -10, "account": {"return_pct": -0.02, "max_drawdown_pct": 0.05}})
    high = score_result({"total_trades": 5, "win_rate": 0.6, "total_pnl_usd": 25, "account": {"return_pct": 0.05, "max_drawdown_pct": 0.02}})
    assert high > low


def test_run_trial_applies_env(monkeypatch):
    history = [
        _snapshot("2026-06-01_100000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_101000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
        _snapshot("2026-06-01_102000", 5050.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
    ]
    cfg = TraderConfig(
        name="test",
        env={"GEX_TRADER_STRICT_FILTERS": "0", "GEX_TRADER_ENTRY_TIME_FILTER": "0"},
        min_confidence=0.4,
        stop_loss=0.05,
        take_profit=0.10,
    )
    row = run_trial(cfg, ticker="SPX", history=history, starting_capital=500.0)
    assert row["name"] == "test"
    assert row["total_trades"] >= 1
    assert row["config"]["env"]["GEX_TRADER_STRICT_FILTERS"] == "0"
