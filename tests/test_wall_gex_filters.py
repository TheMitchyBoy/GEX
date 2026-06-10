"""Wall GEX signal quality filters (opt #5)."""

import pandas as pd

from gex_core.trading.low_gex_signals import wall_entry_quality_ok


def test_wall_entry_quality_blocks_weak_gamma(monkeypatch):
    monkeypatch.setenv("GEX_WALL_SIGNAL_FILTERS", "1")
    monkeypatch.setenv("GEX_WALL_MIN_GAMMA_BN", "0.5")
    monkeypatch.setenv("GEX_WALL_MIN_DRIFT_PTS", "0")
    ok, reason = wall_entry_quality_ok(
        wall_strike=7440.0,
        wall_gamma=-0.2,
        last_wall_strike=None,
    )
    assert ok is False
    assert "below min" in reason.lower()


def test_wall_entry_quality_blocks_short_gamma(monkeypatch):
    monkeypatch.setenv("GEX_WALL_SIGNAL_FILTERS", "1")
    monkeypatch.setenv("GEX_WALL_MIN_GAMMA_BN", "0")
    monkeypatch.setenv("GEX_WALL_BLOCK_SHORT_GAMMA", "1")
    monkeypatch.setenv("GEX_WALL_MIN_DRIFT_PTS", "0")
    ok, reason = wall_entry_quality_ok(
        wall_strike=7440.0,
        wall_gamma=-2.0,
        regime="SHORT gamma",
        last_wall_strike=None,
    )
    assert ok is False
    assert "short-gamma" in reason.lower()


def test_wall_entry_quality_requires_drift(monkeypatch):
    monkeypatch.setenv("GEX_WALL_SIGNAL_FILTERS", "1")
    monkeypatch.setenv("GEX_WALL_MIN_GAMMA_BN", "0")
    monkeypatch.setenv("GEX_WALL_BLOCK_SHORT_GAMMA", "0")
    monkeypatch.setenv("GEX_WALL_MIN_DRIFT_PTS", "10")
    ok, _ = wall_entry_quality_ok(
        wall_strike=7440.0,
        wall_gamma=-2.0,
        last_wall_strike=7440.0,
    )
    assert ok is False
    ok2, _ = wall_entry_quality_ok(
        wall_strike=7460.0,
        wall_gamma=-2.0,
        last_wall_strike=7440.0,
    )
    assert ok2 is True


def test_backtest_skips_weak_gamma(monkeypatch):
    from gex_core.trading.low_gex_backtest import backtest_low_gex_trader

    monkeypatch.setenv("GEX_WALL_INTRADAY_SESSION", "0")
    monkeypatch.setenv("GEX_WALL_ENTRY_TIME_FILTER", "0")
    monkeypatch.setenv("GEX_WALL_SIGNAL_FILTERS", "1")
    monkeypatch.setenv("GEX_WALL_MIN_GAMMA_BN", "1.0")
    monkeypatch.setenv("GEX_WALL_MIN_DRIFT_PTS", "0")
    monkeypatch.setenv("GEX_WALL_BLOCK_SHORT_GAMMA", "0")

    def _row(ts: str, spot: float, strikes: dict[float, float]) -> dict:
        return {
            "ts": ts,
            "spot": spot,
            "strike": pd.Series(strikes, dtype=float),
            "regime": "LONG gamma",
        }

    history = [
        _row("2026-06-02_180000", 7460.0, {7440.0: -0.3, 7460.0: 0.5, 7480.0: 1.0}),
        _row("2026-06-02_180500", 7460.0, {7440.0: -0.4, 7460.0: 0.3, 7480.0: 0.8}),
    ]
    result = backtest_low_gex_trader(
        "SPX",
        history=history,
        starting_capital=5000.0,
        lookback_days=None,
    )
    assert result["total_trades"] == 0
    assert result.get("skipped_wall_weak_gamma", 0) >= 1
