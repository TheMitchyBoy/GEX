"""Tests for prediction intervals, structural attribution, and multi-horizon."""

import pandas as pd

from gex_core.predict import predict_multi_horizon, predict_next_snapshot
from gex_core.structural import attribute_last_move


def _snapshot(ts: str, total_gex: float, spot: float = 5000.0) -> dict:
    strikes = pd.Series(
        [total_gex * 0.25, total_gex * 0.75],
        index=[spot - 50, spot + 50],
        dtype=float,
    )
    return {
        "ts": ts,
        "ts_label": ts,
        "ticker": "SPX",
        "strike": strikes,
        "cumulative": strikes.cumsum(),
        "total_gex": float(total_gex),
        "pos_gex": float(strikes[strikes > 0].sum()),
        "neg_gex": float(strikes[strikes < 0].sum()),
        "gex_std": float(strikes.std()),
        "near_term_ratio": 0.0,
        "surface_peak": 0.0,
        "call_wall": float(strikes.idxmax()),
        "put_wall": float(strikes.idxmin()),
        "gamma_flip": None,
        "regime": "LONG gamma",
        "abs_mean": float(strikes.abs().mean()),
        "spot": spot,
    }


def _history(n: int) -> list[dict]:
    return [
        _snapshot(f"2026-06-{i + 1:02d}_000000", 10.0 + i, spot=5000.0 + i * 5)
        for i in range(n)
    ]


def test_prediction_exposes_interval_and_regime():
    pred = predict_next_snapshot(_history(8), lookback_days=None)
    assert pred is not None
    assert pred["predicted_delta_gex_low"] <= pred["predicted_delta_gex"] <= pred["predicted_delta_gex_high"]
    assert pred["predicted_total_gex_low"] <= pred["predicted_total_gex"] <= pred["predicted_total_gex_high"]
    assert "regime_detail" in pred
    assert "knn_delta_gex" in pred


def test_multi_horizon_returns_requested_horizons():
    results = predict_multi_horizon(_history(12), horizons=(1, 3), lookback_days=None)
    assert 1 in results
    # With enough history the 3-step horizon should also be produced.
    assert 3 in results
    assert results[3]["horizon"] == 3


def test_attribute_last_move_splits_observed_delta():
    hist = _history(3)
    attr = attribute_last_move(hist)
    assert attr is not None
    assert abs(
        attr["observed_delta_gex"] - (attr["spot_component"] + attr["residual_component"])
    ) < 1e-9
