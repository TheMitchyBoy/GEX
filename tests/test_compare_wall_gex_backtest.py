"""compare_wall_gex_backtest defaults and metadata."""

import pandas as pd

from gex_core.trading.low_gex_backtest import compare_wall_gex_backtest


def _synthetic_history(n: int = 6) -> list[dict]:
    rows = []
    for i in range(n):
        spot = 7400.0 + i * 5
        rows.append(
            {
                "ts": f"2026-06-02_13325{i}",
                "spot": spot,
                "regime": "LONG gamma",
                "strike": pd.Series(
                    [-1.5 + i * 0.1, 0.4, 1.2],
                    index=[spot - 10, spot, spot + 10],
                ),
            }
        )
    return rows


def test_compare_near_wall_defaults_no_dedupe():
    history = _synthetic_history(6)
    result = compare_wall_gex_backtest(
        "SPX",
        history=history,
        lookback_days=None,
        window_pct=0.01,
        starting_capital=500.0,
    )
    assert result["near_wall"] is True
    assert result["dedupe_identical_strikes"] is False
    assert result["window_pct"] == 0.01
    assert result["take_profit_pct"] == 0.28
    assert result["max_hold_bars"] == 10
    assert "low_gex" in result and "high_gex" in result
    assert result["recommended_side"] in {"min", "max", "tie", None}


def test_compare_full_wall_can_dedupe_when_explicit():
    history = _synthetic_history(4)
    result = compare_wall_gex_backtest(
        "SPX",
        history=history,
        lookback_days=None,
        window_pct=0.12,
        dedupe_identical_strikes=True,
    )
    assert result["near_wall"] is False
    assert result["dedupe_identical_strikes"] is True
