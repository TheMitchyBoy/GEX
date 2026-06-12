"""Minimal page load: current gamma snapshot only."""

from unittest.mock import patch

import pandas as pd

from gex_core.periscope import build_periscope_context
from gex_core.periscope_api import list_periscope_timestamps, page_minimal_load_enabled


def test_page_minimal_load_enabled_default():
    assert page_minimal_load_enabled() is True


def test_build_periscope_context_minimal_skips_trail_and_previous():
    strikes = pd.Series({5000: 1.0, 5050: -0.5, 5100: 0.3})
    snapshot = {
        "ts": "2026-06-05_193000",
        "ts_label": "2026-06-05 19:30",
        "spot": 5025.0,
        "total_gex": 0.8,
        "regime": "LONG gamma",
        "strike": strikes,
        "cumulative": strikes.cumsum(),
    }
    load_calls: list[bool] = []

    def _load(*_a, minimal=False, **_k):
        load_calls.append(bool(minimal))
        return snapshot

    with (
        patch(
            "gex_core.periscope.list_periscope_timestamps",
            return_value=["2026-06-05_183000", "2026-06-05_193000"],
        ),
        patch("gex_core.periscope.load_periscope_snapshot", side_effect=_load),
        patch("gex_core.periscope.periscope_price_points", return_value=[]),
        patch("gex_core.periscope.uw_api_key", return_value=None),
    ):
        ctx = build_periscope_context(ticker="SPX", selected_ts="2026-06-05_193000", minimal=True)

    assert ctx["minimal"] is True
    assert ctx["exposure_trail"] == []
    assert ctx["previous_exposure"].empty
    assert load_calls and all(load_calls)


def test_list_periscope_timestamps_minimal_uses_index_not_api():
    with (
        patch("gex_core.periscope_api.list_indexed_timestamps_before_date", return_value=["2026-06-04_200000"]),
        patch("gex_core.periscope_api.list_indexed_timestamps_for_date", return_value=["2026-06-05_193000"]),
        patch("gex_core.periscope_api.list_api_intraday_timestamps") as api_ts,
        patch("gex_core.periscope_api.uw_api_configured", return_value=True),
        patch("gex_core.periscope_api.market_today", return_value="2026-06-05"),
    ):
        ts = list_periscope_timestamps("SPX", minimal=True)
    api_ts.assert_not_called()
    assert ts == ["2026-06-04_200000", "2026-06-05_193000"]
