from unittest.mock import patch

import pandas as pd

from gex_core.periscope_api import (
    IntradayDayCache,
    list_periscope_timestamps,
    load_periscope_snapshot,
    snapshot_from_uw_entry,
    utc_today,
)


def test_list_periscope_timestamps_merges_history_and_api_today():
    historical = ["2026-06-04_032228", "2026-06-05_021908"]
    api_today = ["2026-06-06_101000", "2026-06-06_102000"]
    with (
        patch("gex_core.periscope_api.list_indexed_timestamps_before_date", return_value=historical),
        patch("gex_core.periscope_api.uw_api_configured", return_value=True),
        patch("gex_core.periscope_api.list_api_intraday_timestamps", return_value=api_today),
        patch("gex_core.periscope_api.utc_today", return_value="2026-06-06"),
    ):
        timestamps = list_periscope_timestamps("SPX", api_key="test-key")
    assert timestamps == historical + api_today


def test_load_periscope_snapshot_uses_api_cache_for_today():
    ts = "2026-06-06_101000"
    strike = pd.Series({5000: 1.0, 5050: -0.5})
    cache = IntradayDayCache(
        market_date="2026-06-06",
        timestamps=[ts],
        snapshots={
            ts: {
                "ts": ts,
                "strike": strike,
                "spot": 5025.0,
                "total_gex": 0.5,
            }
        },
        fetched_at=1.0,
    )
    with (
        patch("gex_core.periscope_api.utc_today", return_value="2026-06-06"),
        patch("gex_core.periscope_api.should_use_api_for_date", return_value=True),
        patch("gex_core.periscope_api.fetch_intraday_day_cache", return_value=cache),
    ):
        row = load_periscope_snapshot("SPX", ts, api_key="test-key", market_date="2026-06-06")
    assert row is not None
    assert row["ts"] == ts
    assert float(row["spot"]) == 5025.0


def test_snapshot_from_uw_entry_builds_strike_profile():
    strike = pd.Series({5000: 2.0, 5100: -1.0})
    uw_entry = {
        "spot": 5050.0,
        "agg": type("Agg", (), {"gex_by_strike": strike, "total_gex_bn": 1.0})(),
    }
    row = snapshot_from_uw_entry("SPX", uw_entry, ts="2026-06-06_120000")
    assert row["ts"] == "2026-06-06_120000"
    assert row["spot"] == 5050.0
    assert not row["strike"].empty
