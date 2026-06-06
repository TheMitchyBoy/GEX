import pandas as pd

from gex_core.periscope import (
    build_periscope_context,
    build_timeline_navigation,
    resolve_selected_timestamp,
    slices_for_date,
)
from gex_core.market_exposure_agent import analyze_market_exposure
from gex_core.charts import make_periscope_exposure_chart, make_mm_positions_chart


def test_build_periscope_context_from_history():
    from unittest.mock import patch

    strikes = pd.Series({5000: 1.0, 5050: -0.5, 5100: 0.3})
    snapshot = {
        "ts": "2026-06-05_021908",
        "ts_label": "2026-06-05 02:19",
        "spot": 5025.0,
        "total_gex": 0.8,
        "regime": "LONG gamma",
        "gamma_flip": 5000.0,
        "call_wall": 5100.0,
        "put_wall": 4950.0,
        "strike": strikes,
        "cumulative": strikes.cumsum(),
    }
    with (
        patch(
            "gex_core.periscope.list_periscope_timestamps",
            return_value=["2026-06-05_021908"],
        ),
        patch("gex_core.periscope.load_periscope_snapshot", return_value=snapshot),
        patch("gex_core.periscope.periscope_price_points", return_value=[]),
        patch("gex_core.periscope.list_periscope_dates", return_value=["2026-06-05"]),
        patch("gex_core.periscope.uw_api_key", return_value=None),
    ):
        ctx = build_periscope_context(
            ticker="SPX",
            selected_ts="2026-06-05_021908",
        )
    assert ctx["spot"] == 5025.0
    assert ctx["regime"] == "LONG gamma"
    assert not ctx["exposure_series"].empty


def test_market_exposure_agent_returns_who_what():
    strikes = pd.Series({5000: 2.0, 5050: -1.0, 5100: 3.0})
    result = analyze_market_exposure(
        ticker="SPX",
        spot=5025.0,
        gex_by_strike=strikes,
        total_gex_bn=4.0,
        gamma_flip=5000.0,
    )
    assert result["who"]
    assert result["whom"]
    assert result["what"]
    assert result["narrative"]


def test_resolve_selected_timestamp_by_date():
    timestamps = [
        "2026-06-04_032228",
        "2026-06-05_021908",
        "2026-06-05_031908",
    ]
    assert resolve_selected_timestamp(timestamps, date="2026-06-05") == "2026-06-05_031908"
    assert resolve_selected_timestamp(timestamps, ts="2026-06-04_032228") == "2026-06-04_032228"


def test_build_timeline_navigation_rewind_target():
    timestamps = [
        "2026-06-05_021908",
        "2026-06-05_031908",
        "2026-06-05_041908",
    ]
    nav = build_timeline_navigation(timestamps, "2026-06-05_031908")
    assert nav["prev_ts"] == "2026-06-05_021908"
    assert nav["next_ts"] == "2026-06-05_041908"
    assert nav["selected_date"] == "2026-06-05"
    assert len(slices_for_date(timestamps, "2026-06-05")) == 3


def test_periscope_charts_render_json():
    strikes = pd.Series({5000: 1.0, 5050: -0.5})
    chart = make_periscope_exposure_chart(strikes, spot=5025.0, exposure_type="gamma")
    assert chart is not None
    assert "data" in chart
    pos = make_mm_positions_chart(
        {"net_call_delta_bn": 1.0, "net_put_delta_bn": -0.5, "net_call_gex_bn": 2.0, "net_put_gex_bn": -1.0}
    )
    assert pos is not None
