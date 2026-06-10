"""Low-GEX walk-forward backtest."""

import pandas as pd

from gex_core.trading.low_gex_backtest import backtest_low_gex_trader


def _row(ts: str, spot: float, strikes: dict[float, float]) -> dict:
    return {
        "ts": ts,
        "spot": spot,
        "strike": pd.Series(strikes, dtype=float),
        "regime": "LONG gamma",
    }


def test_backtest_low_gex_trader_runs_on_synthetic_history():
    history = [
        _row("2026-06-02_100000", 7460.0, {7440.0: -2.0, 7460.0: 0.5, 7480.0: 1.0}),
        _row("2026-06-02_101000", 7455.0, {7440.0: -2.5, 7460.0: 0.3, 7480.0: 0.8}),
        _row("2026-06-02_102000", 7448.0, {7440.0: -3.0, 7460.0: 0.1, 7480.0: 0.5}),
        _row("2026-06-02_103000", 7442.0, {7440.0: -2.0, 7460.0: -0.2, 7480.0: 0.2}),
    ]
    result = backtest_low_gex_trader(
        "SPX",
        history=history,
        starting_capital=500.0,
        lookback_days=None,
    )
    assert result["strategy"] == "low_gex"
    assert result["snapshots"] == 4
