import numpy as np
import pandas as pd

from gex_core.features import estimate_gamma_flip, top_strike_concentration


def test_gamma_flip_crossing():
    idx = [4700, 4800, 4900]
    cumulative = pd.Series([-1.0, -0.2, 0.5], index=idx)
    flip = estimate_gamma_flip(cumulative)
    assert flip is not None
    assert 4800 < flip < 4900


def test_top_strike_concentration():
    strike = pd.Series([1.0, 0.1, 0.1], index=[4800, 4810, 4820])
    conc = top_strike_concentration(strike, top_n=1)
    assert conc > 0.5
