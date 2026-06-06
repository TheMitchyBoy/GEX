import json

import pandas as pd

from gex_core.charts import (
    make_0dte_movement_chart,
    make_gex_profile_chart,
    make_spx_price_chart,
    make_timeline_chart,
)


def test_gex_profile_chart_uses_tight_spot_window_and_bar_width():
    strikes = pd.Series(
        [1.0, -0.5, 2.0, -1.0, 0.7],
        index=[4700, 4800, 4900, 5000, 5100],
        dtype=float,
    )

    payload = json.loads(make_gex_profile_chart(strikes, "SPX", spot=4900.0))

    bar = payload["data"][0]
    assert bar["width"] > 1
    assert payload["layout"]["xaxis"]["range"][0] <= 4700
    assert payload["layout"]["xaxis"]["range"][1] >= 5100
    assert payload["layout"]["bargap"] == 0.0
    xaxis = payload["layout"]["xaxis"]
    assert xaxis.get("tickmode") == "array" or xaxis.get("dtick") == 100


def test_chart_strike_series_pins_spot_level():
    from gex_core.charts import _chart_strike_series

    series = pd.Series([1.0, -1.0, 2.0], index=[7200.0, 7350.0, 7500.0])
    window = _chart_strike_series(series, 7383.0, window_pct=0.01, max_bars=2, pin_levels=(7383.0,))
    assert 7350.0 in window.index


def test_0dte_movement_chart_compares_same_day_snapshots():
    previous = {
        "ts": "2026-06-05_140000",
        "ts_label": "2026-06-05 14:00:00",
        "spot": 5000,
        "strike": pd.Series({4975.0: 1.0, 5000.0: -2.0, 5025.0: 0.5}),
    }
    current = {
        "ts": "2026-06-05_141000",
        "ts_label": "2026-06-05 14:10:00",
        "spot": 5005,
        "strike": pd.Series({4975.0: 1.5, 5000.0: -1.0, 5025.0: -0.5}),
    }

    payload = json.loads(make_0dte_movement_chart(current, previous, "SPX", spot=5005))

    bar = payload["data"][0]
    assert bar["name"] == "ΔGEX since prior same-day snapshot"
    assert bar["y"] == [0.5, 1.0, -1.0]
    assert bar["width"] > 1


def test_spx_price_chart_prefers_live_points():
    points = [
        {"ts": "2026-06-05T13:30:00", "close": 5000.0},
        {"ts": "2026-06-05T14:00:00", "close": 5012.5},
    ]
    payload = json.loads(make_spx_price_chart(points, ticker="SPX"))
    line = payload["data"][0]
    assert line["y"] == [5000.0, 5012.5]
    assert "Unusual Whales" in payload["layout"]["title"]["text"]
    # Latest price annotated as current marker.
    assert payload["data"][1]["y"] == [5012.5]


def test_spx_price_chart_falls_back_to_snapshot_spots():
    history = [
        {"ts_label": "2026-06-04 00:00:00", "spot": 4990.0},
        {"ts_label": "2026-06-05 00:00:00", "spot": 5010.0},
    ]
    payload = json.loads(make_spx_price_chart(None, history=history, ticker="SPX"))
    assert payload["data"][0]["y"] == [4990.0, 5010.0]
    assert "snapshots" in payload["layout"]["title"]["text"]


def test_spx_price_chart_returns_none_without_data():
    assert make_spx_price_chart(None, history=[]) is None


def test_timeline_chart_plots_spot_and_max_positive_gamma_strike():
    history = [
        {
            "ts_label": "2026-06-04 15:00:00",
            "spot": 4990.0,
            "pos_gamma_peak_strike": 5000.0,
            "gamma_flip": 4985.0,
            "call_wall": 5000.0,
            "put_wall": 4925.0,
            "total_gex": 1.2,
            "near_term_ratio": 0.4,
            "regime": "LONG gamma",
        },
        {
            "ts_label": "2026-06-05 15:00:00",
            "spot": 5010.0,
            "pos_gamma_peak_strike": 5025.0,
            "gamma_flip": 5005.0,
            "call_wall": 5025.0,
            "put_wall": 4950.0,
            "total_gex": 0.8,
            "near_term_ratio": 0.35,
            "regime": "LONG gamma",
        },
    ]

    payload = json.loads(make_timeline_chart(history, "SPX"))
    names = [trace["name"] for trace in payload["data"] if trace.get("name")]
    assert "SPX spot" in names
    assert "Max +γ strike" in names
    spot = next(trace for trace in payload["data"] if trace.get("name") == "SPX spot")
    peak = next(trace for trace in payload["data"] if trace.get("name") == "Max +γ strike")
    assert spot["y"] == [4990.0, 5010.0]
    assert peak["y"] == [5000.0, 5025.0]
    assert "Max +γ Strike" in payload["layout"]["title"]["text"]
