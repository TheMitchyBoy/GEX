"""Low-GEX walk-forward backtest."""

import pandas as pd

from gex_core.trading.low_gex_backtest import backtest_low_gex_trader

# Export keys are UTC; 18:00 UTC ≈ 14:00 ET (inside RTH entry window).
_DAY = "2026-06-02"
_IN = "180000"


def _row(ts: str, spot: float, strikes: dict[float, float]) -> dict:
    return {
        "ts": ts,
        "spot": spot,
        "strike": pd.Series(strikes, dtype=float),
        "regime": "LONG gamma",
    }


def _rth(hhmmss: str, spot: float, strikes: dict[float, float]) -> dict:
    return _row(f"{_DAY}_{hhmmss}", spot, strikes)


def _disable_wall_session_filters(monkeypatch):
    monkeypatch.setenv("GEX_WALL_INTRADAY_SESSION", "0")
    monkeypatch.setenv("GEX_WALL_ENTRY_TIME_FILTER", "0")
    monkeypatch.setenv("GEX_WALL_SIGNAL_FILTERS", "0")


def test_backtest_low_gex_trader_runs_on_synthetic_history(monkeypatch):
    _disable_wall_session_filters(monkeypatch)
    history = [
        _rth("180000", 7460.0, {7440.0: -2.0, 7460.0: 0.5, 7480.0: 1.0}),
        _rth("180500", 7455.0, {7440.0: -2.5, 7460.0: 0.3, 7480.0: 0.8}),
        _rth("181000", 7448.0, {7440.0: -3.0, 7460.0: 0.1, 7480.0: 0.5}),
        _rth("181500", 7442.0, {7440.0: -2.0, 7460.0: -0.2, 7480.0: 0.2}),
    ]
    result = backtest_low_gex_trader(
        "SPX",
        history=history,
        starting_capital=500.0,
        lookback_days=None,
    )
    assert result["strategy"] == "low_gex"
    assert result["snapshots"] == 4


def test_backtest_low_gex_reenter_each_bar_opens_every_snapshot(monkeypatch):
    _disable_wall_session_filters(monkeypatch)
    history = [
        _rth("180000", 7460.0, {7440.0: -2.0, 7460.0: 0.5, 7480.0: 1.0}),
        _rth("180500", 7455.0, {7440.0: -2.5, 7460.0: 0.3, 7480.0: 0.8}),
        _rth("181000", 7448.0, {7440.0: -3.0, 7460.0: 0.1, 7480.0: 0.5}),
        _rth("181500", 7442.0, {7440.0: -2.0, 7460.0: -0.2, 7480.0: 0.2}),
    ]
    for row in history:
        row["interval_minutes"] = 5
    result = backtest_low_gex_trader(
        "SPX",
        history=history,
        starting_capital=5000.0,
        lookback_days=None,
        reenter_each_bar=True,
    )
    assert result["total_trades"] == 3
    assert result.get("by_exit_reason", {}).get("bar_rotation", 0) >= 2


def test_wall_max_hold_bars_defaults_to_eight(monkeypatch):
    monkeypatch.delenv("GEX_WALL_MAX_HOLD_BARS", raising=False)
    monkeypatch.delenv("GEX_WALL_MAX_HOLD_MINUTES", raising=False)
    from gex_core.trading.config import wall_max_hold_bars

    assert wall_max_hold_bars() == 8


def test_backtest_low_gex_default_sl_tp(monkeypatch):
    _disable_wall_session_filters(monkeypatch)
    monkeypatch.delenv("GEX_WALL_STOP_LOSS_PCT", raising=False)
    monkeypatch.delenv("GEX_WALL_TAKE_PROFIT_PCT", raising=False)
    history = [
        _rth("180000", 7460.0, {7440.0: -2.0, 7460.0: 0.5, 7480.0: 1.0}),
        _rth("180500", 7460.0, {7440.0: -2.5, 7460.0: 0.3, 7480.0: 0.8}),
    ]
    result = backtest_low_gex_trader("SPX", history=history, starting_capital=5000.0, lookback_days=None)
    assert result["stop_loss_pct"] == 0.03
    assert result["take_profit_pct"] == 0.20
    assert result.get("max_hold_bars") == 8


def test_backtest_low_gex_skips_late_session_entries():
    history = [
        _rth("193500", 7460.0, {7440.0: -2.0, 7460.0: 0.5, 7480.0: 1.0}),
        _rth("194000", 7460.0, {7440.0: -2.5, 7460.0: 0.3, 7480.0: 0.8}),
    ]
    result = backtest_low_gex_trader(
        "SPX",
        history=history,
        starting_capital=5000.0,
        lookback_days=None,
        intraday_session=True,
        entry_time_filter=True,
    )
    assert result["total_trades"] == 0
    assert result.get("skipped_filters", 0) >= 1


def test_backtest_low_gex_wall_shift_closes_on_new_wall(monkeypatch):
    _disable_wall_session_filters(monkeypatch)
    history = [
        _rth("180000", 7460.0, {7440.0: -2.0, 7460.0: 0.5, 7480.0: 1.0}),
        _rth("180500", 7460.0, {7440.0: -2.5, 7460.0: 0.3, 7480.0: 0.8}),
        _rth("181000", 7460.0, {7440.0: -0.2, 7460.0: -3.0, 7480.0: 0.5}),
    ]
    result = backtest_low_gex_trader(
        "SPX",
        history=history,
        starting_capital=5000.0,
        lookback_days=None,
        reenter_on_shift=True,
    )
    assert result.get("by_exit_reason", {}).get("wall_shift", 0) >= 1


def test_backtest_low_gex_max_hold_exits(monkeypatch):
    _disable_wall_session_filters(monkeypatch)
    monkeypatch.setenv("GEX_WALL_MAX_HOLD_BARS", "1")
    history = [
        _rth("180000", 7460.0, {7440.0: -2.0, 7460.0: 0.5, 7480.0: 1.0}),
        _rth("180500", 7460.0, {7440.0: -2.5, 7460.0: 0.3, 7480.0: 0.8}),
        _rth("181000", 7460.0, {7440.0: -2.5, 7460.0: 0.3, 7480.0: 0.8}),
    ]
    for row in history:
        row["interval_minutes"] = 5
    result = backtest_low_gex_trader(
        "SPX",
        history=history,
        starting_capital=5000.0,
        lookback_days=None,
    )
    assert result.get("by_exit_reason", {}).get("max_hold", 0) >= 1


def test_backtest_low_gex_stop_loss_and_take_profit(monkeypatch):
    _disable_wall_session_filters(monkeypatch)
    history = [
        _rth("180000", 7460.0, {7440.0: -2.0, 7460.0: 0.5, 7480.0: 1.0}),
        _rth("180500", 7460.0, {7440.0: -2.5, 7460.0: 0.3, 7480.0: 0.8}),
        _rth("181000", 7700.0, {7440.0: -3.0, 7460.0: 0.1, 7480.0: 0.5}),
    ]
    result = backtest_low_gex_trader(
        "SPX",
        history=history,
        starting_capital=5000.0,
        lookback_days=None,
        stop_loss=0.05,
        take_profit=0.40,
        reenter_each_bar=False,
    )
    stop_trades = [t for t in result["trades"] if t["exit_reason"] == "stop_loss"]
    assert stop_trades
    assert stop_trades[0]["pnl_pct"] == -0.05
