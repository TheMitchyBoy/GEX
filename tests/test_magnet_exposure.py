"""Dashboard charts use raw spot-exposures/strike net gamma (no magnet transform)."""

from unittest.mock import patch

import pandas as pd

from gex_core.periscope import _greek_exposure_from_df, build_periscope_context
from gex_core.spot_exposure import spot_exposure_net_series


def test_exposure_series_uses_spot_oi_not_greek_magnet():
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

    exposure = ctx["exposure_series"]
    chart = ctx["magnet_exposure_series"]
    assert float(exposure.loc[7570.0]) == -1.0
    assert float(chart.loc[7570.0]) == -1.0
    assert ctx["gamma_flip"] is None


def test_greek_exposure_from_df_reads_gex_column():
    surface = pd.DataFrame({"strike": [7500.0, 7510.0], "GEX": [1.5, -2.0]})
    series = _greek_exposure_from_df(surface, "gamma")
    assert float(series.loc[7500.0]) == 1.5
    assert float(series.loc[7510.0]) == -2.0


def test_chart_series_matches_spot_oi_without_uw_entry():
    spot_df = pd.DataFrame(
        {
            "strike": [7440.0, 7450.0, 7460.0, 7470.0, 7480.0],
            "call_gamma_oi": [1e9, 1e9, 2e9, 3e9, 2e9],
            "put_gamma_oi": [-3e9, -3e9, -2.5e9, -1e9, -0.5e9],
        }
    )
    spot_df["net_gamma_oi"] = spot_df["call_gamma_oi"] + spot_df["put_gamma_oi"]
    spot_df["net_gamma_oi_bn"] = spot_df["net_gamma_oi"] / 1e9
    spot_strike = spot_exposure_net_series(spot_df, "gamma")

    snapshot = {
        "ts": "2026-06-08_151806",
        "spot": 7460.0,
        "strike": spot_strike,
        "spot_exposures_df": spot_df,
    }

    with (
        patch("gex_core.periscope.list_periscope_timestamps", return_value=["2026-06-08_151806"]),
        patch("gex_core.periscope.load_periscope_snapshot", return_value=snapshot),
        patch("gex_core.periscope.periscope_price_points", return_value=[]),
        patch("gex_core.periscope.list_periscope_dates", return_value=["2026-06-08"]),
        patch("gex_core.periscope.uw_api_key", return_value="test"),
        patch("gex_core.periscope.should_use_api_for_date", return_value=True),
    ):
        ctx = build_periscope_context(
            ticker="SPX",
            selected_ts="2026-06-08_151806",
            uw_entry=None,
        )

    assert float(ctx["exposure_series"].loc[7450.0]) == -2.0
    assert float(ctx["magnet_exposure_series"].loc[7450.0]) == -2.0
    assert ctx["gamma_flip"] is None
