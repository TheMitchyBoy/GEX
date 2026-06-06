"""Periscope falls back to greek exposure when spot grid misses spot."""

from unittest.mock import patch

import pandas as pd

from gex_core.periscope import build_periscope_context


def test_periscope_uses_greek_when_spot_exposures_miss_spot():
    spot_df = pd.DataFrame(
        {
            "strike": [6800.0, 6900.0, 7000.0, 7100.0, 7180.0],
            "call_gamma_oi": [1e9, 1e9, 2e9, 1e9, 1e9],
            "put_gamma_oi": [-0.5e9, -0.5e9, -1e9, -0.5e9, -0.5e9],
        }
    )
    spot_df["net_gamma_oi"] = spot_df["call_gamma_oi"] + spot_df["put_gamma_oi"]
    spot_df["net_gamma_oi_bn"] = spot_df["net_gamma_oi"] / 1e9

    greek_df = pd.DataFrame(
        {
            "strike": [7300.0, 7350.0, 7380.0, 7400.0, 7450.0],
            "net_gex": [1.0, 2.0, 3.0, 2.5, 1.5],
            "call_gex": [1.0, 2.0, 3.0, 2.5, 1.5],
            "put_gex": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    class FakeAgg:
        gex_by_strike = pd.Series(dtype=float)
        surface_data = pd.DataFrame()

    FakeAgg.gex_by_strike.attrs = {
        "spot_exposures_df": spot_df,
        "greek_exposure_df": greek_df,
    }

    uw_entry = {"spot": 7383.0, "agg": FakeAgg(), "spot_gamma_bn": 2.0, "fetched_at": "now"}
    snapshot = {
        "ts": "2026-06-06_041933",
        "ts_label": "2026-06-06 04:19",
        "spot": 7383.0,
        "total_gex": 2.0,
        "regime": "LONG gamma",
        "strike": pd.Series(dtype=float),
        "spot_exposures_df": spot_df,
    }

    with (
        patch("gex_core.periscope.list_periscope_timestamps", return_value=["2026-06-06_041933"]),
        patch("gex_core.periscope.load_periscope_snapshot", return_value=snapshot),
        patch("gex_core.periscope.periscope_price_points", return_value=[]),
        patch("gex_core.periscope.list_periscope_dates", return_value=["2026-06-06"]),
        patch("gex_core.periscope.uw_api_key", return_value="test"),
    ):
        ctx = build_periscope_context(
            ticker="SPX",
            selected_ts="2026-06-06_041933",
            uw_entry=uw_entry,
        )

    exposure = ctx["exposure_series"]
    assert 7380.0 in exposure.index
    assert exposure.index.max() >= 7380.0

    window = ctx["exposure_window"]
    assert window.index.min() >= 7300.0
    assert window.index.max() <= 7450.0
