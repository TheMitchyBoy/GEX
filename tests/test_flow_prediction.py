"""Tests for option-flow overlay on GEX predictions."""

import math
from pathlib import Path

import pandas as pd

from gex_core.predict import apply_flow_to_prediction, load_flow_predictions


def test_load_flow_predictions_returns_strike_breakdown(tmp_path: Path):
    feed = tmp_path / "flow.jsonl"
    feed.write_text(
        '{"option":"SPX260620C04800000","gamma":0.00012,"quantity":50,"side":"buy","spot":4800}\n'
    )
    flow = load_flow_predictions(feed, spot=4800.0)
    assert flow["event_count"] == 1
    assert flow["predicted_flow_delta_gex_bn"] != 0.0
    assert 4800.0 in flow["flow_by_strike_bn"]


def test_apply_flow_to_prediction_adjusts_totals_and_strikes():
    pred = {
        "predicted_delta_gex": 0.1,
        "predicted_total_gex": 1.0,
        "predicted_strike": pd.Series({4800.0: 0.05, 4900.0: 0.02}),
    }
    flow = {
        "event_count": 2,
        "predicted_flow_delta_gex_bn": 0.01,
        "flow_by_strike_bn": {4800.0: 0.008, 5000.0: 0.002},
        "top_signals": [],
    }
    out = apply_flow_to_prediction(pred, flow)
    expected_weight = math.log1p(2) / math.log(101.0)
    expected_flow_delta = 0.01 * expected_weight
    assert out["flow_blend_weight"] == expected_weight
    assert out["raw_flow_delta_gex"] == 0.01
    assert out["predicted_delta_gex"] == 0.1 + expected_flow_delta
    assert out["predicted_total_gex"] == 1.0 + expected_flow_delta
    assert out["flow_delta_gex"] == expected_flow_delta
    assert out["predicted_strike"][4800.0] == 0.05 + (0.008 * expected_weight)
    assert out["predicted_strike"][5000.0] == 0.002 * expected_weight


def test_apply_flow_skips_when_no_events():
    pred = {"predicted_delta_gex": 0.1, "predicted_total_gex": 1.0}
    flow = {"event_count": 0, "predicted_flow_delta_gex_bn": 0.0, "flow_by_strike_bn": {}}
    out = apply_flow_to_prediction(pred, flow)
    assert out is pred
    assert "flow_delta_gex" not in out
