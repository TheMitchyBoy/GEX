"""Tests for advisor confidence Monte Carlo optimization."""

from __future__ import annotations

import pandas as pd

from gex_core.trading.backtest import backtest_auto_trader
from gex_core.trading.backtest_agent import (
    format_confidence_monte_carlo_reply,
    run_agent_confidence_monte_carlo,
    user_wants_confidence_monte_carlo,
)
from gex_core.trading.monte_carlo_confidence import (
    _confidence_grid,
    run_confidence_monte_carlo,
    summarize_confidence_monte_carlo,
)


def _snapshot(ts: str, spot: float, strikes: dict[float, float]) -> dict:
    return {
        "ts": ts,
        "spot": spot,
        "strike": pd.Series(strikes),
        "regime": "LONG gamma",
        "gamma_flip": spot * 0.995,
        "flow_net_delta_gex_bn": 0.5,
        "interval_minutes": 10,
        "is_cpi_day": False,
        "is_nfp_day": False,
        "is_fomc_week": False,
    }


def test_user_wants_confidence_monte_carlo_detects_phrases():
    assert user_wants_confidence_monte_carlo("Run a Monte Carlo confidence sweep")
    assert user_wants_confidence_monte_carlo("optimize advisor confidence for best ROI")
    assert not user_wants_confidence_monte_carlo("What is the gamma flip?")


def test_confidence_grid_includes_production():
    grid = _confidence_grid(min_conf_start=0.50, min_conf_stop=0.50, strong_levels=[0.80])
    names = [c.name for c in grid]
    assert "production" in names
    assert "min_0.50_strong_0.80" in names


def test_backtest_skips_low_confidence(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_MAGNET_PROGRESS", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_ENTRY_CONFIDENCE", "0.99")

    history = [
        _snapshot("2026-06-01_100000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_101000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
        _snapshot("2026-06-01_102000", 5050.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
    ]
    result = backtest_auto_trader(
        "SPX",
        history=history,
        stop_loss=0.05,
        take_profit=0.10,
        max_open=1,
    )
    assert result.get("skipped_low_confidence", 0) >= 1
    assert result["total_trades"] == 0


def test_run_confidence_monte_carlo_ranks_trials(monkeypatch):
    history = [
        _snapshot("2026-06-01_100000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_101000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
        _snapshot("2026-06-01_102000", 5050.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
    ]
    monkeypatch.setattr(
        "gex_core.trading.monte_carlo_confidence._build_history_impl",
        lambda *a, **k: history,
    )
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_MAGNET_PROGRESS", "0")

    summary = run_confidence_monte_carlo(
        ticker="SPX",
        lookback_days=7,
        max_snapshots=50,
        min_conf_start=0.0,
        min_conf_stop=0.0,
        min_conf_step=0.05,
        strong_levels=[0.80],
    )
    assert summary["trials_run"] >= 2
    assert summary.get("best_roi") is not None or summary.get("best") is not None

    compact = summarize_confidence_monte_carlo(summary)
    assert compact["trials_run"] == summary["trials_run"]
    assert "best_roi" in compact


def test_run_agent_confidence_monte_carlo(monkeypatch):
    history = [
        _snapshot("2026-06-01_100000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_101000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
        _snapshot("2026-06-01_102000", 5050.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
    ]
    monkeypatch.setattr(
        "gex_core.trading.monte_carlo_confidence._build_history_impl",
        lambda *a, **k: history,
    )
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")

    result = run_agent_confidence_monte_carlo(
        "SPX",
        lookback_days=7,
        max_snapshots=50,
        min_conf_start=0.0,
        min_conf_stop=0.0,
        strong_levels=[0.80],
    )
    assert result.get("trials_run", 0) >= 1


def test_format_confidence_monte_carlo_reply():
    text = format_confidence_monte_carlo_reply(
        {
            "snapshots": 100,
            "trials_run": 12,
            "trials_profitable": 3,
            "best_roi": {
                "min_entry_confidence": 0.55,
                "strong_confidence": 0.80,
                "total_trades": 5,
                "win_rate": 0.6,
                "return_pct": 0.12,
            },
        }
    )
    assert "0.55" in text
    assert "12" in text


def test_api_monte_carlo_confidence_endpoint(monkeypatch):
    from web_app import APP

    history = [
        _snapshot("2026-06-01_100000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_101000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
    ]
    monkeypatch.setattr(
        "gex_core.trading.monte_carlo_confidence._build_history_impl",
        lambda *a, **k: history,
    )
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")

    client = APP.test_client()
    response = client.get(
        "/api/agent/monte-carlo-confidence?lookback_days=1&max_snapshots=20"
        "&min_conf_start=0&min_conf_stop=0&strong_levels=0.80"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("trials_run", 0) >= 1
