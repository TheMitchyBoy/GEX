"""Tests for Unusual Whales context bundle assembly."""

import pandas as pd

from gex_core.pipeline import GexAggregates
from gex_core.uw_context_bundle import build_uw_context_bundle, bundle_token_estimate


def _sample_greek_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strike": [5000.0, 5050.0, 5100.0, 5150.0],
            "call_gex": [1.0, 2.0, 3.0, 1.5],
            "put_gex": [-0.5, -1.0, -0.8, -2.0],
            "net_gex": [0.5, 1.0, 2.2, -0.5],
            "call_delta": [0.1, 0.2, 0.3, 0.15],
            "put_delta": [-0.05, -0.1, -0.08, -0.2],
            "call_charm": [0.01, 0.02, 0.03, 0.01],
            "put_charm": [-0.01, -0.02, -0.01, -0.03],
            "call_vanna": [0.05, 0.06, 0.07, 0.04],
            "put_vanna": [-0.03, -0.04, -0.02, -0.05],
        }
    )


def _sample_spot_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strike": [5040.0, 5050.0, 5060.0],
            "price": [5050.0, 5050.0, 5050.0],
            "call_gamma_oi": [1e9, 2e9, 1.5e9],
            "put_gamma_oi": [-0.5e9, -1e9, -0.8e9],
            "call_gamma_vol": [0.2e9, 0.3e9, 0.25e9],
            "put_gamma_vol": [-0.1e9, -0.15e9, -0.12e9],
        }
    )


def _sample_agg() -> GexAggregates:
    greek_df = _sample_greek_df()
    spot_df = _sample_spot_df()
    spot_df["net_gamma_oi"] = spot_df["call_gamma_oi"] + spot_df["put_gamma_oi"]
    spot_df["net_gamma_oi_bn"] = spot_df["net_gamma_oi"] / 1e9

    gex_by_strike = pd.Series(greek_df["net_gex"].values, index=greek_df["strike"].values)
    gex_by_strike.attrs["greek_exposure_df"] = greek_df
    gex_by_strike.attrs["spot_exposures_df"] = spot_df

    return GexAggregates(
        gex_by_strike=gex_by_strike,
        gex_by_expiration=pd.Series({pd.Timestamp("2026-06-06"): 2.5, pd.Timestamp("2026-06-13"): 1.2}),
        cumulative_gex=gex_by_strike.cumsum(),
        surface_data=pd.DataFrame({"strike": greek_df["strike"], "GEX": greek_df["net_gex"]}),
        total_gex_bn=float(gex_by_strike.sum()),
    )


def test_build_uw_context_bundle_includes_all_sections():
    agg = _sample_agg()
    history = [
        {"ts_label": "2026-06-05_120000", "spot": 5040.0, "total_gex": 2.0, "regime": "LONG gamma"},
        {"ts_label": "2026-06-05_130000", "spot": 5050.0, "total_gex": 2.2, "regime": "LONG gamma"},
    ]
    knn = {"predicted_delta_gex_bn": 0.15, "predicted_regime": "LONG gamma", "confidence": 0.62}

    bundle = build_uw_context_bundle(
        ticker="SPX",
        spot=5050.0,
        agg=agg,
        gamma_flip=5025.0,
        spot_gamma_bn=2.1,
        history=history,
        knn_prediction=knn,
        fetch_extras=False,
    )

    assert bundle["ticker"] == "SPX"
    assert bundle["spot"] == 5050.0
    assert bundle["summary"]["total_gex_bn"] == agg.total_gex_bn
    assert len(bundle["greek_exposure_by_strike"]) > 0
    assert len(bundle["spot_exposures_by_strike"]) > 0
    assert len(bundle["gex_by_expiration"]) == 2
    assert bundle["knn_forecast"]["predicted_delta_gex_bn"] == 0.15
    assert len(bundle["snapshot_history"]) == 2
    assert "net_charm_bn" in bundle["extended_features"] or bundle["extended_features"]
    assert bundle_token_estimate(bundle) > 50


def test_build_uw_context_bundle_empty_agg_graceful():
    empty = GexAggregates(
        gex_by_strike=pd.Series(dtype=float),
        gex_by_expiration=pd.Series(dtype=float),
        cumulative_gex=pd.Series(dtype=float),
        surface_data=pd.DataFrame(),
        total_gex_bn=0.0,
    )
    bundle = build_uw_context_bundle(
        ticker="SPX",
        spot=5000.0,
        agg=empty,
        fetch_extras=False,
    )
    assert bundle["greek_exposure_by_strike"] == []
    assert bundle["summary"]["total_gex_bn"] == 0.0
