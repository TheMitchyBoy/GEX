"""Tests for strike filtering near spot."""

import pandas as pd

from gex_core.strike_filter import (
    filter_strikes_for_storage,
    resolve_storage_strike_profile,
    strikes_bracket_spot,
)


def test_strikes_bracket_spot_false_for_far_chain():
    far = pd.Series({200.0: -0.01, 400.0: -0.02, 600.0: -0.01})
    assert strikes_bracket_spot(far, 7554.0) is False


def test_filter_strikes_for_storage_keeps_atm_window():
    strikes = pd.Series(
        {
            200.0: -0.01,
            7400.0: 1.5,
            7500.0: 2.0,
            7600.0: -1.0,
            12000.0: 0.01,
        }
    )
    filtered = filter_strikes_for_storage(strikes, 7550.0, window_pct=0.05, min_strikes=3)
    assert 200.0 not in filtered.index
    assert 12000.0 not in filtered.index
    assert 7500.0 in filtered.index


def test_resolve_storage_uses_greek_when_spot_rows_misaligned():
    spot_profile = pd.Series({200.0: -1e-5, 400.0: -2e-5, 600.0: -1e-5})
    greek_df = pd.DataFrame(
        {
            "strike": [7400.0, 7500.0, 7600.0],
            "call_gex": [1.0, 2.0, 1.5],
            "put_gex": [-0.5, -1.0, -0.8],
            "net_gex": [0.5, 1.0, 0.7],
        }
    )
    resolved, source = resolve_storage_strike_profile(spot_profile, spot=7550.0, greek_df=greek_df)
    assert source.startswith("greek_exposure")
    assert 200.0 not in resolved.index
    assert 7500.0 in resolved.index
