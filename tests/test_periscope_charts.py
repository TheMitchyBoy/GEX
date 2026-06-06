import json

import pandas as pd
import pytest

from gex_core.periscope_charts import (
    build_periscope_charts,
    cumulative_exposure_chart,
    dealer_positions_chart,
    exposure_by_strike_chart,
    exposure_change_chart,
    session_price_chart,
    strike_ladder_chart,
)


def test_session_price_chart_highlights_slice():
    points = [
        {"ts": "2026-06-06 09:30", "close": 5300.0},
        {"ts": "2026-06-06 09:40", "close": 5310.0},
    ]
    payload = json.loads(session_price_chart(points, spot=5310.0, highlight_label="2026-06-06 09:40"))
    assert payload["layout"]["height"] == 380
    assert len(payload["data"][0]["y"]) == 2


def test_exposure_by_strike_has_prior_dots_and_vertical_bars():
    series = pd.Series([0.5, -0.3, 0.8], index=[7350.0, 7380.0, 7410.0])
    prev = pd.Series([0.4, -0.2, 0.7], index=[7350.0, 7380.0, 7410.0])
    payload = json.loads(
        exposure_by_strike_chart(
            series,
            spot=7380.0,
            previous=prev,
            gamma_flip=7385.0,
            call_wall=7410.0,
            put_wall=7350.0,
        )
    )
    assert payload["data"][0]["type"] == "bar"
    assert len(payload["data"]) == 2
    assert payload["data"][1]["mode"] == "markers"


def test_exposure_change_chart_shows_delta():
    current = pd.Series([1.0, -0.5], index=[7380.0, 7400.0])
    previous = pd.Series([0.8, -0.4], index=[7380.0, 7400.0])
    payload = json.loads(exposure_change_chart(current, previous, spot=7385.0))
    assert payload["data"][0]["type"] == "bar"
    assert payload["data"][0]["y"][0] == pytest.approx(0.2)


def test_cumulative_exposure_chart_renders_line():
    series = pd.Series([1.0, -0.5, 0.3], index=[7380.0, 7400.0, 7420.0])
    payload = json.loads(cumulative_exposure_chart(series, spot=7390.0, gamma_flip=7400.0))
    assert payload["data"][0]["mode"] == "lines"
    assert payload["data"][0]["fill"] == "tozeroy"


def test_dealer_positions_chart_vertical_bars():
    payload = json.loads(
        dealer_positions_chart(
            {
                "net_call_gex_bn": 1.2,
                "net_put_gex_bn": -0.8,
                "net_call_delta_bn": 0.5,
                "net_put_delta_bn": -0.3,
            }
        )
    )
    assert payload["data"][0]["type"] == "bar"
    assert payload["data"][0].get("orientation") != "h"


def test_strike_ladder_chart_horizontal_bars_from_center():
    series = pd.Series([0.8, -0.5, 0.3, -0.2], index=[7380.0, 7390.0, 7400.0, 7410.0])
    payload = json.loads(strike_ladder_chart(series, spot=7395.0, gamma_flip=7390.0))
    assert payload["data"][0]["orientation"] == "h"
    assert payload["data"][0]["x"][0] == 0.8
    assert payload["data"][0]["x"][1] == -0.5
    spot_trace = next(t for t in payload["data"] if t.get("uid") == "spot-line")
    assert spot_trace["y"] == [7395.0, 7395.0]
    assert any(a.get("x") == 0 for a in payload["layout"].get("annotations", []))


def test_build_periscope_charts_bundle():
    profile = pd.Series([1.0, -1.0], index=[7380.0, 7400.0])
    bundle = build_periscope_charts(
        ticker="SPX",
        exposure_type="gamma",
        spot=7385.0,
        exposure_profile=profile,
        exposure_extended=profile,
        previous_exposure=profile * 0.9,
        price_points=[{"ts": "09:30", "close": 7385.0}],
        highlight_label="09:30",
        mm_positions={"net_call_gex_bn": 1.0, "net_put_gex_bn": -1.0, "net_call_delta_bn": 0.0, "net_put_delta_bn": 0.0},
        gamma_flip=7390.0,
    )
    assert bundle.ladder is not None
    assert bundle.price is not None
    assert bundle.exposures is not None
    assert bundle.change is not None
    assert bundle.cumulative is not None
    assert bundle.positions is not None
