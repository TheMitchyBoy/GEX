import json

import pandas as pd

from gex_core.periscope_charts import (
    build_periscope_charts,
    exposure_profile_chart,
    positions_chart,
    price_chart,
)


def test_price_chart_highlights_slice():
    points = [
        {"ts": "2026-06-06 09:30", "close": 5300.0},
        {"ts": "2026-06-06 09:40", "close": 5310.0},
    ]
    payload = json.loads(price_chart(points, spot=5310.0, highlight_label="2026-06-06 09:40"))
    assert payload["layout"]["height"] == 420
    assert len(payload["data"][0]["y"]) == 2


def test_exposure_profile_has_white_prior_dots():
    series = pd.Series([0.5, -0.3, 0.8], index=[7350.0, 7380.0, 7410.0])
    prev = pd.Series([0.4, -0.2, 0.7], index=[7350.0, 7380.0, 7410.0])
    payload = json.loads(exposure_profile_chart(series, spot=7380.0, previous=prev))
    assert payload["layout"]["yaxis"]["type"] == "category"
    assert len(payload["data"]) == 2
    assert payload["data"][1]["mode"] == "markers"


def test_positions_chart_horizontal_bars():
    payload = json.loads(
        positions_chart(
            {
                "net_call_gex_bn": 1.2,
                "net_put_gex_bn": -0.8,
                "net_call_delta_bn": 0.5,
                "net_put_delta_bn": -0.3,
            }
        )
    )
    assert payload["data"][0]["orientation"] == "h"


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
    )
    assert bundle.price is not None
    assert bundle.exposures is not None
    assert bundle.exposures_extended is not None
    assert bundle.positions is not None
