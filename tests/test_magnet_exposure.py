"""Magnet map uses greek exposure (shows +γ below spot), not spot-exposures OI only."""

from unittest.mock import patch

import pandas as pd

from gex_core.periscope import build_periscope_context


def test_magnet_exposure_prefers_greek_over_spot_oi():
    spot_df = pd.DataFrame(
        {
            "strike": [7560.0, 7570.0, 7580.0, 7590.0, 7600.0],
            "call_gamma_oi": [1e9, 1e9, 2e9, 3e9, 2e9],
            "put_gamma_oi": [-2e9, -2e9, -1.5e9, -1e9, -0.5e9],
        }
    )
    spot_df["net_gamma_oi"] = spot_df["call_gamma_oi"] + spot_df["put_gamma_oi"]
    spot_df["net_gamma_oi_bn"] = spot_df["net_gamma_oi"] / 1e9

    greek_df = pd.DataFrame(
        {
            "strike": [7560.0, 7570.0, 7580.0, 7590.0, 7600.0],
            "net_gex": [-1.0, 2.5, 1.0, 3.0, 2.0],
            "call_gex": [0.5, 2.5, 1.0, 3.0, 2.0],
            "put_gex": [-1.5, 0.0, 0.0, 0.0, 0.0],
        }
    )

    class FakeAgg:
        gex_by_strike = pd.Series(dtype=float)
        surface_data = pd.DataFrame()

    FakeAgg.gex_by_strike.attrs = {
        "spot_exposures_df": spot_df,
        "greek_exposure_df": greek_df,
    }

    uw_entry = {"spot": 7580.0, "agg": FakeAgg(), "spot_gamma_bn": 1.0, "fetched_at": "now"}
    snapshot = {
        "ts": "2026-06-08_145806",
        "ts_label": "2026-06-08 14:58",
        "spot": 7580.0,
        "total_gex": 1.0,
        "regime": "LONG gamma",
        "strike": pd.Series(dtype=float),
        "spot_exposures_df": spot_df,
    }

    with (
        patch("gex_core.periscope.list_periscope_timestamps", return_value=["2026-06-08_145806"]),
        patch("gex_core.periscope.load_periscope_snapshot", return_value=snapshot),
        patch("gex_core.periscope.periscope_price_points", return_value=[]),
        patch("gex_core.periscope.list_periscope_dates", return_value=["2026-06-08"]),
        patch("gex_core.periscope.uw_api_key", return_value="test"),
    ):
        ctx = build_periscope_context(
            ticker="SPX",
            selected_ts="2026-06-08_145806",
            uw_entry=uw_entry,
        )

    spot_series = ctx["exposure_series"]
    magnet = ctx["magnet_exposure_series"]
    below_spot = magnet[magnet.index < 7580.0]
    assert (spot_series[spot_series.index < 7580.0] > 0).sum() == 0
    assert (below_spot > 0).sum() >= 1
    assert float(magnet.loc[7570.0]) == 2.5
