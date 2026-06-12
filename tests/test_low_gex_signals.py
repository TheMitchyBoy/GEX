"""Low-GEX strike signal: call/put toward minimum gamma wall."""

import pandas as pd

from gex_core.trading.low_gex_signals import (
    compute_low_gex_signal,
    compute_near_wall_gex_signal,
    compute_wall_gex_signal,
)


def test_low_gex_below_spot_buys_puts():
    exposure = pd.Series(
        [-2.0, -0.5, 0.3, 1.0],
        index=[7440.0, 7450.0, 7460.0, 7470.0],
    )
    pack = compute_low_gex_signal(exposure, spot=7460.0)
    assert pack["available"] is True
    rec = pack["recommended"]
    assert rec["option_type"] == "put"
    assert rec["strike"] == 7440.0
    assert rec["gamma_bn"] == -2.0


def test_high_gex_targets_max_positive_strike():
    from gex_core.trading.low_gex_signals import compute_high_gex_signal

    exposure = pd.Series(
        [-1.0, 0.5, 2.5, 1.0],
        index=[7440.0, 7450.0, 7460.0, 7470.0],
    )
    pack = compute_high_gex_signal(exposure, spot=7460.0)
    assert pack["available"] is True
    rec = pack["recommended"]
    assert rec["strike"] == 7460.0
    assert rec["gamma_bn"] == 2.5
    assert rec["option_type"] == "call"


def test_low_gex_above_spot_buys_calls():
    exposure = pd.Series(
        [0.5, 1.0, -1.5, -0.2],
        index=[7440.0, 7450.0, 7470.0, 7480.0],
    )
    pack = compute_low_gex_signal(exposure, spot=7460.0)
    assert pack["available"] is True
    rec = pack["recommended"]
    assert rec["option_type"] == "call"
    assert rec["strike"] == 7470.0


def test_low_gex_unavailable_without_data():
    pack = compute_low_gex_signal(pd.Series(dtype=float), spot=5000.0)
    assert pack["available"] is False


def test_extreme_target_picks_highest_abs_gamma():
    exposure = pd.Series(
        [-1.0, 0.5, 3.0, 1.0],
        index=[7440.0, 7450.0, 7460.0, 7470.0],
    )
    pack = compute_wall_gex_signal(exposure, spot=7460.0, target="extreme")
    rec = pack["recommended"]
    assert rec["strike"] == 7460.0
    assert rec["gamma_bn"] == 3.0
    assert rec["option_type"] == "call"
    assert rec["signal_type"] == "max_gamma_strike"


def test_extreme_target_picks_lowest_when_more_negative():
    exposure = pd.Series(
        [-4.0, -0.5, 1.0, 0.8],
        index=[7440.0, 7450.0, 7460.0, 7470.0],
    )
    pack = compute_wall_gex_signal(exposure, spot=7460.0, target="extreme")
    rec = pack["recommended"]
    assert rec["strike"] == 7440.0
    assert rec["gamma_bn"] == -4.0
    assert rec["option_type"] == "put"
    assert rec["signal_type"] == "min_gamma_strike"


def test_near_wall_signal_matches_extreme():
    exposure = pd.Series(
        [-4.0, -0.5, 1.0, 0.8],
        index=[7440.0, 7450.0, 7460.0, 7470.0],
    )
    pack = compute_near_wall_gex_signal(exposure, spot=7460.0, window_pct=0.01)
    rec = pack["recommended"]
    assert rec["strike"] == 7440.0
    assert rec["option_type"] == "put"
