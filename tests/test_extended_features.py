import pandas as pd

from gex_core.event_calendar import event_calendar_features
from gex_core.extended_features import (
    merge_extended_features,
    summarize_flow_feed,
    summarize_greek_exposure_df,
    summarize_spot_exposures_df,
)
from gex_core.features import snapshot_feature_vector


def test_summarize_greek_exposure_df():
    df = pd.DataFrame(
        {
            "strike": [4800.0, 4850.0],
            "call_charm": [1.0, 0.5],
            "put_charm": [-0.5, -0.2],
            "call_vanna": [2.0, 1.0],
            "put_vanna": [-1.0, -0.5],
            "call_delta": [10.0, 5.0],
            "put_delta": [-8.0, -4.0],
        }
    )
    out = summarize_greek_exposure_df(df, spot=4800.0)
    assert out["net_charm_bn"] == 0.8
    assert out["net_vanna_bn"] == 1.5
    assert out["net_delta_bn"] == 3.0
    assert out["charm_at_spot_bn"] == 0.5


def test_summarize_spot_exposures_df():
    df = pd.DataFrame(
        {
            "strike": [4800.0, 4850.0],
            "call_gamma_oi": [2e9, 1e9],
            "put_gamma_oi": [-1e9, -0.5e9],
            "call_gamma_vol": [1e9, 0.5e9],
            "put_gamma_vol": [-0.5e9, -0.25e9],
        }
    )
    out = summarize_spot_exposures_df(df)
    assert out["gamma_oi_bn"] == 1.5
    assert out["gamma_vol_bn"] == 0.75
    assert out["gamma_oi_vol_ratio"] > 0


def test_merge_extended_features_and_vector_size():
    metrics = {"spot": 4800.0, "total_gex": 1.0, "pos_gex": 1.0, "neg_gex": 0.0, "gex_std": 0.1, "near_term_ratio": 0.5}
    merge_extended_features(
        metrics,
        greek_df=pd.DataFrame({"strike": [4800.0], "call_charm": [1.0], "put_charm": [0.0], "call_vanna": [1.0], "put_vanna": [0.0]}),
        market_date="2026-06-05",
        vol_regime={"vix_level": 18.0, "vix9d_level": 17.0, "iv_rank": 0.4, "skew_proxy": 0.05, "expected_move_pct": 0.011},
    )
    vec = snapshot_feature_vector(metrics)
    assert vec.shape[0] == 50
    assert metrics["vix_level"] == 18.0
    assert metrics["is_nfp_day"] == 1.0


def test_summarize_flow_feed(tmp_path):
    feed = tmp_path / "flow.jsonl"
    feed.write_text(
        '{"option":"SPX260620C04800000","gamma":0.0001,"quantity":10,"side":"buy","spot":4800}\n'
        '{"option":"SPX260620P04700000","gamma":0.0001,"quantity":5,"side":"sell","spot":4800}\n'
    )
    out = summarize_flow_feed(feed)
    assert out["flow_event_count"] == 2.0
    assert out["flow_buy_ratio"] == 0.5


def test_event_calendar_features_flags_fomc_week():
    out = event_calendar_features("2026-06-16")
    assert out["is_fomc_week"] == 1.0
