from unittest.mock import patch

import pandas as pd

from gex_core.periscope import (
    build_periscope_context,
    build_timeline_navigation,
    resolve_selected_timestamp,
    slices_for_date,
)
from gex_core.market_exposure_agent import analyze_market_exposure


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
        "2026-06-05_183000",
        "2026-06-05_193000",
    ]
    with patch("gex_core.periscope.list_indexed_timestamps_for_date", return_value=timestamps):
        assert resolve_selected_timestamp(timestamps, date="2026-06-05") == "2026-06-05_193000"
    assert resolve_selected_timestamp(timestamps, ts="2026-06-05_183000") == "2026-06-05_183000"


def test_slices_for_date_uses_market_session_not_utc_prefix():
    from gex_core.periscope import slices_for_date

    timestamps = ["2026-06-06_012248", "2026-06-06_031908"]
    with patch("gex_core.periscope.list_indexed_timestamps_for_date", return_value=[]):
        day_slices = slices_for_date(timestamps, "2026-06-05")
    assert day_slices == timestamps


def test_build_timeline_navigation_rewind_target():
    timestamps = [
        "2026-06-05_183000",
        "2026-06-05_193000",
        "2026-06-05_203000",
    ]
    with patch("gex_core.periscope.list_indexed_timestamps_for_date", return_value=timestamps):
        nav = build_timeline_navigation(timestamps, "2026-06-05_193000")
        assert nav["prev_ts"] == "2026-06-05_183000"
        assert nav["next_ts"] == "2026-06-05_203000"
        assert nav["selected_date"] == "2026-06-05"
        assert len(slices_for_date(timestamps, "2026-06-05")) == 3
