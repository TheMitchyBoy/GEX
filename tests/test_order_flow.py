"""Tests for options order-flow dashboard payload."""

from pathlib import Path

from gex_core.order_flow import build_order_flow_snapshot, compute_flow_trend


def test_compute_flow_trend_bullish_when_net_delta_positive():
    flow = {
        "event_count": 5,
        "predicted_flow_delta_gex_bn": 0.08,
        "top_signals": [
            {"direction": "up", "score": 0.6},
            {"direction": "up", "score": 0.4},
        ],
    }
    trend = compute_flow_trend(flow, buy_ratio=0.7)
    assert trend["direction"] == "up"
    assert trend["label"] == "Bullish"
    assert trend["bullish_strikes"] == 2


def test_compute_flow_trend_neutral_when_no_events():
    trend = compute_flow_trend({"event_count": 0, "top_signals": []})
    assert trend["direction"] == "neutral"
    assert trend["label"] == "No flow"


def test_build_order_flow_snapshot_from_sample_feed(tmp_path: Path):
    feed = tmp_path / "flow.jsonl"
    feed.write_text(
        '{"option":"SPX260620C04800000","gamma":0.00012,"quantity":50,"side":"buy","spot":4800}\n'
        '{"option":"SPX260627P04900000","gamma":0.00008,"quantity":40,"side":"sell","spot":4800}\n'
    )
    out = build_order_flow_snapshot(feed, spot=4800.0, snapshot={"flow_buy_ratio": 0.6})
    assert out["event_count"] == 2
    assert out["trend"]["direction"] in {"up", "down", "neutral"}
    assert isinstance(out["top_signals"], list)
