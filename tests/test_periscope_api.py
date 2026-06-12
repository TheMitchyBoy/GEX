from unittest.mock import patch

import pandas as pd

from gex_core.periscope_api import (
    IntradayDayCache,
    _store_day_cache,
    clear_periscope_api_cache,
    list_periscope_timestamps,
    load_periscope_snapshot,
    snapshot_from_uw_entry,
)


def test_day_cache_evicts_oldest_entry_when_full(monkeypatch):
    monkeypatch.setattr("gex_core.periscope_api._MAX_DAY_CACHE_ENTRIES", 2)
    clear_periscope_api_cache()

    first = IntradayDayCache(market_date="2026-06-04", fetched_at=1.0)
    second = IntradayDayCache(market_date="2026-06-05", fetched_at=2.0)
    third = IntradayDayCache(market_date="2026-06-06", fetched_at=3.0)
    _store_day_cache(("SPX", "2026-06-04"), first)
    _store_day_cache(("SPX", "2026-06-05"), second)
    _store_day_cache(("SPX", "2026-06-06"), third)

    from gex_core.periscope_api import _day_cache

    assert len(_day_cache) == 2
    assert ("SPX", "2026-06-04") not in _day_cache
    assert ("SPX", "2026-06-06") in _day_cache
    clear_periscope_api_cache()


def test_list_periscope_timestamps_merges_history_and_api_today():
    historical = ["2026-06-04_032228", "2026-06-05_021908"]
    api_today = ["2026-06-06_101000", "2026-06-06_102000"]
    with (
        patch("gex_core.periscope_api.list_indexed_timestamps_before_date", return_value=historical),
        patch("gex_core.periscope_api.uw_api_configured", return_value=True),
        patch("gex_core.periscope_api.list_api_intraday_timestamps", return_value=api_today),
        patch("gex_core.periscope_api.market_today", return_value="2026-06-06"),
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
        patch("gex_core.periscope_api.market_today", return_value="2026-06-06"),
        patch("gex_core.periscope_api.should_use_api_for_date", return_value=True),
        patch("gex_core.periscope_api.fetch_intraday_day_cache", return_value=cache),
    ):
        row = load_periscope_snapshot("SPX", ts, api_key="test-key", market_date="2026-06-06")
    assert row is not None
    assert row["ts"] == ts
    assert float(row["spot"]) == 5025.0


def test_snapshot_from_uw_entry_uses_spot_exposures_profile():
    strike = pd.Series({5000: 2.0, 5100: -1.0})
    spot_df = pd.DataFrame(
        {
            "strike": [5000.0, 5100.0],
            "call_gamma_oi": [3.0e9, 1.0e9],
            "put_gamma_oi": [-1.0e9, -2.0e9],
            "price": [5050.0, 5050.0],
        }
    )
    gex_by_strike = strike.copy()
    gex_by_strike.attrs = {"spot_exposures_df": spot_df}
    uw_entry = {
        "spot": 5050.0,
        "spot_gamma_bn": -87.0,
        "agg": type("Agg", (), {"gex_by_strike": gex_by_strike, "total_gex_bn": 1.0})(),
    }
    row = snapshot_from_uw_entry("SPX", uw_entry, ts="2026-06-06_120000")
    assert row["ts"] == "2026-06-06_120000"
    assert row["spot"] == 5050.0
    assert float(row["total_gex"]) == -87.0
    assert float(row["strike"].loc[5000.0]) == 2.0
    assert row.get("spot_exposures_df") is not None
