"""ATM-local gamma flip estimation."""

import pandas as pd

from gex_core.exports import load_strike_series
from gex_core.features import estimate_gamma_flip, gamma_flip_from_profile
from pathlib import Path


def test_gamma_flip_from_profile_uses_atm_window_not_full_chain():
    strike = load_strike_series(Path("data/exports/SPX_gex_by_strike_2026-06-05_021908.csv"))
    spot = 7580.0
    full_flip = estimate_gamma_flip(strike.cumsum())
    atm_flip = gamma_flip_from_profile(strike, spot)
    assert full_flip is not None
    assert atm_flip is not None
    assert abs(atm_flip - spot) < abs(full_flip - spot)
    assert atm_flip < full_flip


def test_gamma_flip_from_profile_small_series():
    series = pd.Series([1.0, -2.0, 3.0], index=[100.0, 105.0, 110.0])
    flip = gamma_flip_from_profile(series, 107.0)
    assert flip is not None
    assert 100.0 < flip < 110.0
