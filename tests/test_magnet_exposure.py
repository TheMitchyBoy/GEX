"""Magnet map uses greek exposure (shows +γ below spot), not spot-exposures OI only."""

from unittest.mock import patch

import pandas as pd

from gex_core.periscope import (
    _greek_exposure_from_df,
    _magnet_gamma_from_call_put,
    build_periscope_context,
)
from gex_core.spot_exposure import spot_exposure_net_series


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


def test_greek_exposure_from_df_reads_gex_column():
    surface = pd.DataFrame({"strike": [7500.0, 7510.0], "GEX": [1.5, -2.0]})
    series = _greek_exposure_from_df(surface, "gamma")
    assert float(series.loc[7500.0]) == 1.5
    assert float(series.loc[7510.0]) == -2.0


def test_magnet_uses_greek_strike_without_uw_entry():
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
    greek_strike = pd.Series(
        [-1.0, 2.5, 1.0, 3.0, 2.0],
        index=[7440.0, 7450.0, 7460.0, 7470.0, 7480.0],
        dtype=float,
    )

    snapshot = {
        "ts": "2026-06-08_151806",
        "spot": 7460.0,
        "strike": spot_strike,
        "spot_exposures_df": spot_df,
        "greek_strike": greek_strike,
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

    magnet = ctx["magnet_exposure_series"]
    assert (magnet[magnet.index < 7460.0] > 0).sum() >= 1
    assert float(magnet.loc[7450.0]) == 2.5
    assert (ctx["exposure_series"][ctx["exposure_series"].index < 7460.0] > 0).sum() == 0


def test_magnet_uses_surface_data_when_greek_df_missing():
    surface = pd.DataFrame({"strike": [7440.0, 7450.0, 7460.0], "GEX": [-1.0, 2.5, 1.0]})
    spot_df = pd.DataFrame(
        {
            "strike": [7440.0, 7450.0, 7460.0],
            "call_gamma_oi": [1e9, 1e9, 2e9],
            "put_gamma_oi": [-3e9, -3e9, -2.5e9],
        }
    )
    spot_df["net_gamma_oi"] = spot_df["call_gamma_oi"] + spot_df["put_gamma_oi"]
    spot_df["net_gamma_oi_bn"] = spot_df["net_gamma_oi"] / 1e9

    class FakeAgg:
        gex_by_strike = pd.Series(surface.set_index("strike")["GEX"])
        surface_data = surface

    FakeAgg.gex_by_strike.attrs = {"spot_exposures_df": spot_df}

    snapshot = {
        "ts": "2026-06-08_151806",
        "spot": 7460.0,
        "strike": spot_exposure_net_series(spot_df, "gamma"),
        "spot_exposures_df": spot_df,
    }

    with (
        patch("gex_core.periscope.list_periscope_timestamps", return_value=["2026-06-08_151806"]),
        patch("gex_core.periscope.load_periscope_snapshot", return_value=snapshot),
        patch("gex_core.periscope.periscope_price_points", return_value=[]),
        patch("gex_core.periscope.list_periscope_dates", return_value=["2026-06-08"]),
        patch("gex_core.periscope.uw_api_key", return_value="test"),
    ):
        ctx = build_periscope_context(
            ticker="SPX",
            selected_ts="2026-06-08_151806",
            uw_entry={"spot": 7460.0, "agg": FakeAgg()},
        )

    magnet = ctx["magnet_exposure_series"]
    assert float(magnet.loc[7450.0]) == 2.5


def test_magnet_gamma_shows_dominant_call_when_net_cancels_at_7430():
    greek_df = pd.DataFrame(
        {
            "strike": [7420.0, 7430.0, 7460.0],
            "net_gex": [-1.226612, 0.006224, 1.206460],
            "call_gex": [2.624939, 4.630437, 3.573736],
            "put_gex": [-3.851552, -4.624214, -2.367276],
        }
    )
    magnet = _magnet_gamma_from_call_put(greek_df, spot=7460.8)
    assert float(magnet.loc[7430.0]) == 4.630437
    assert float(magnet.loc[7420.0]) < 0
    assert float(magnet.loc[7460.0]) == 1.206460


def test_magnet_context_uses_call_gex_for_cancelled_7430():
    greek_df = pd.DataFrame(
        {
            "strike": [7420.0, 7430.0, 7460.0, 7480.0],
            "net_gex": [-1.226612, 0.006224, 1.206460, 2.0],
            "call_gex": [2.624939, 4.630437, 3.573736, 2.0],
            "put_gex": [-3.851552, -4.624214, -2.367276, -1.0],
        }
    )
    snapshot = {
        "ts": "2026-06-08_151806",
        "spot": 7460.8,
        "strike": pd.Series(dtype=float),
        "greek_exposure_df": greek_df,
    }
    with (
        patch("gex_core.periscope.list_periscope_timestamps", return_value=["2026-06-08_151806"]),
        patch("gex_core.periscope.load_periscope_snapshot", return_value=snapshot),
        patch("gex_core.periscope.periscope_price_points", return_value=[]),
        patch("gex_core.periscope.list_periscope_dates", return_value=["2026-06-08"]),
        patch("gex_core.periscope.uw_api_key", return_value="test"),
    ):
        ctx = build_periscope_context(ticker="SPX", selected_ts="2026-06-08_151806")
    magnet = ctx["magnet_exposure_series"]
    assert float(magnet.loc[7430.0]) == 4.630437
