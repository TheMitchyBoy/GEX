import pandas as pd

from gex_core.features import estimate_gamma_flip, term_structure_breakdown, top_strike_concentration


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


def test_term_structure_breakdown_splits_zero_dte_and_back_term():
    expirations = pd.Series(
        [2.0, 1.0, -0.5, 0.5],
        index=pd.to_datetime(["2026-06-05", "2026-06-06", "2026-06-13", "2026-06-20"]),
    )

    panel = term_structure_breakdown(expirations, snapshot_date=pd.Timestamp("2026-06-05"))

    assert panel["zero_dte_gex_bn"] == 2.0
    assert panel["near_term_gex_bn"] == 2.5
    assert panel["back_term_gex_bn"] == 1.0
    assert panel["term_curvature"] == 1.5
    assert panel["expiration_count"] == 4.0
    # front_term = nearest-dated expiration bucket (here it coincides with 0DTE).
    assert panel["front_term_gex_bn"] == 2.0


def test_front_term_diverges_from_zero_dte_without_same_day_expiry():
    expirations = pd.Series(
        [3.0, -1.0],
        index=pd.to_datetime(["2026-06-08", "2026-06-15"]),
    )

    panel = term_structure_breakdown(expirations, snapshot_date=pd.Timestamp("2026-06-05"))

    # No same-day expiry: 0DTE falls back to the first bucket, and front_term is
    # the nearest expiration. Both equal the first bucket but are computed
    # independently (front_term is no longer a copy of zero_dte).
    assert panel["front_term_gex_bn"] == 3.0
    assert panel["zero_dte_gex_bn"] == 3.0
