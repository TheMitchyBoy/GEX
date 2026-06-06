"""Tests for ATM strike selection."""

import pandas as pd

from gex_core.features import select_atm_strike_series, spot_covers_strike_grid


def test_select_atm_strike_series_window_around_spot():
    series = pd.Series(
        {7000: 1.0, 7100: 2.0, 7200: -1.0, 7300: 3.0, 7400: -2.0, 7500: 1.5, 8000: 4.0},
    )
    atm = select_atm_strike_series(series, 7383.0, window_pct=0.03, min_strikes=3, max_strikes=10)
    assert atm.index.min() >= 7000
    assert atm.index.max() <= 7600
    assert 7300 in atm.index or 7400 in atm.index


def test_select_atm_strike_series_nearest_when_window_sparse():
    series = pd.Series({6000: 1.0, 6500: -1.0, 7000: 2.0, 7700: 3.0, 7800: -2.0})
    atm = select_atm_strike_series(series, 7383.0, window_pct=0.01, min_strikes=3, max_strikes=5)
    assert 7700 in atm.index
    assert 7000 not in atm.index or len(atm) <= 5


def test_spot_covers_strike_grid():
    series = pd.Series({7200: 1.0, 7300: 2.0, 7400: 3.0})
    assert spot_covers_strike_grid(series, 7350.0)
    assert not spot_covers_strike_grid(series, 7600.0)
