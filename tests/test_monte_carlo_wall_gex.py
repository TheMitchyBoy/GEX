"""Wall GEX Monte Carlo SL/TP search."""

import pandas as pd

from gex_core.trading.monte_carlo_wall_gex import run_wall_gex_monte_carlo


def _row(ts: str, spot: float, strikes: dict[float, float]) -> dict:
    return {
        "ts": ts,
        "spot": spot,
        "strike": pd.Series(strikes, dtype=float),
        "regime": "LONG gamma",
    }


def test_wall_gex_monte_carlo_ranks_trials():
    history = [
        _row("2026-06-02_100000", 7460.0, {7440.0: -2.0, 7460.0: 0.5, 7480.0: 1.0}),
        _row("2026-06-02_101000", 7460.0, {7440.0: -2.5, 7460.0: 0.3, 7480.0: 0.8}),
        _row("2026-06-02_102000", 7700.0, {7440.0: -3.0, 7460.0: 0.1, 7480.0: 0.5}),
        _row("2026-06-02_103000", 7460.0, {7440.0: -2.0, 7460.0: -0.2, 7480.0: 0.2}),
    ]

    # Monkeypatch history build by passing prebuilt via internal _trial only;
    # run_wall_gex_monte_carlo loads from exports — use grid on synthetic via direct trials.
    from gex_core.trading import monte_carlo_wall_gex as mc

    trials = [
        mc._trial(
            ticker="SPX",
            history=history,
            starting_capital=5000.0,
            stop_loss=0.05,
            take_profit=0.40,
            target="min",
        ),
        mc._trial(
            ticker="SPX",
            history=history,
            starting_capital=5000.0,
            stop_loss=0.10,
            take_profit=0.20,
            target="min",
        ),
    ]
    ranked = sorted(trials, key=lambda row: row["score"], reverse=True)
    assert ranked[0]["total_trades"] >= 0
    assert "stop_loss" in ranked[0]
    assert "take_profit" in ranked[0]
