"""Tests for market-context features and regime detection."""

import math

from gex_core.market_features import (
    attach_market_features,
    latest_spot_return,
    realized_volatility,
)
from gex_core.regime import classify_regime, model_blend_weight, volatility_bucket


def _hist(spots, totals=None):
    totals = totals or [1.0] * len(spots)
    return [{"spot": s, "total_gex": t} for s, t in zip(spots, totals)]


def test_realized_vol_zero_for_flat_series():
    assert realized_volatility(_hist([100, 100, 100, 100])) == 0.0


def test_realized_vol_positive_for_moving_series():
    rv = realized_volatility(_hist([100, 102, 99, 103, 98, 104]))
    assert rv > 0.0


def test_latest_spot_return_matches_last_step():
    ret = latest_spot_return(_hist([100, 110]))
    assert abs(ret - 0.1) < 1e-9


def test_attach_market_features_is_causal():
    hist = _hist([100, 102, 101, 105, 103])
    attach_market_features(hist)
    # First row has no prior return -> zero realized vol / return.
    assert hist[0]["realized_vol"] == 0.0
    assert hist[0]["spot_return"] == 0.0
    # Later rows are populated.
    assert hist[-1]["realized_vol"] >= 0.0
    assert "vix_level" in hist[-1]


def test_regime_label_combines_gamma_and_vol():
    hist = _hist([100, 101, 99, 103, 97, 105, 96], totals=[5, 5, 5, 5, 5, 5, 5])
    attach_market_features(hist)
    regime = classify_regime(hist)
    assert regime["gamma"] == "LONG gamma"
    assert regime["volatility"] in {"low-vol", "mid-vol", "high-vol", "unknown"}


def test_model_blend_weight_leans_on_knn_in_high_vol():
    assert model_blend_weight("high-vol") < model_blend_weight("low-vol")
