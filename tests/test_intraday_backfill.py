from unittest.mock import patch

import pandas as pd
import pytest

from gex_core.intraday_backfill import (
    backfill_intraday_minutes,
    minute_row_total_gex_bn,
    sample_intraday_rows,
    scale_strike_profile,
    uw_time_to_export_ts,
)


def test_uw_time_to_export_ts_utc():
    assert uw_time_to_export_ts("2026-06-05T14:30:00Z") == "2026-06-05_143000"


def test_minute_row_total_gex_bn_from_aggregate_column():
    row = pd.Series({"gamma_per_one_percent_move_oi": 2.5e9})
    assert minute_row_total_gex_bn(row) == pytest.approx(2.5)


def test_minute_row_total_gex_bn_from_call_put_oi():
    row = pd.Series({"call_gamma_oi": 1.5e9, "put_gamma_oi": -0.5e9})
    assert minute_row_total_gex_bn(row) == pytest.approx(1.0)


def test_sample_intraday_rows_every_ten_minutes():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-06-05T14:30:00Z",
                    "2026-06-05T14:35:00Z",
                    "2026-06-05T14:39:00Z",
                    "2026-06-05T14:40:00Z",
                ],
                utc=True,
            ),
            "price": [5000.0, 5001.0, 5002.0, 5003.0],
        }
    )
    sampled = sample_intraday_rows(df, 10)
    assert len(sampled) == 2
    # Last row in each 10-minute bucket is kept.
    assert uw_time_to_export_ts(sampled.iloc[0]["time"]) == "2026-06-05_143900"
    assert uw_time_to_export_ts(sampled.iloc[1]["time"]) == "2026-06-05_144000"


def test_scale_strike_profile_matches_target_total():
    strike = pd.Series([1.0, 2.0, 3.0], index=[100.0, 101.0, 102.0])
    scaled = scale_strike_profile(strike, 12.0)
    assert float(scaled.sum()) == pytest.approx(12.0)


@patch("gex_core.intraday_backfill.cached_cross_asset_returns", return_value={})
@patch("gex_core.intraday_backfill.cached_vol_regime", return_value={})
@patch("gex_core.uw_loader.fetch_uw_spot_exposures")
@patch("gex_core.uw_loader.fetch_uw_spot_exposures_intraday")
def test_backfill_intraday_minutes_writes_snapshots(
    mock_intraday,
    mock_spot_strike,
    _mock_vol,
    _mock_cross,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("gex_core.intraday_backfill.EXPORT_DIR", tmp_path)
    monkeypatch.setenv("GEX_INDEX_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("GEX_MIN_STRIKE_COUNT", "2")
    mock_intraday.return_value = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-06-05T14:30:00Z", "2026-06-05T14:40:00Z"], utc=True),
            "price": [5000.0, 5001.0],
            "gamma_per_one_percent_move_oi": [1.0e9, 2.0e9],
        }
    )
    mock_spot_strike.return_value = pd.DataFrame(
        {
            "strike": [4900.0, 5000.0],
            "call_gamma_oi": [1.0e9, 1.0e9],
            "put_gamma_oi": [-0.5e9, -0.5e9],
            "price": [5000.0, 5000.0],
        }
    )

    saved = backfill_intraday_minutes(
        "SPX", "2026-06-05", export_dir=tmp_path, interval_minutes=10
    )
    assert saved == 2
    assert (tmp_path / "SPX_gex_by_strike_2026-06-05_143000.csv").exists()
    assert (tmp_path / "SPX_summary_2026-06-05_144000.json").exists()

    saved_again = backfill_intraday_minutes("SPX", "2026-06-05", export_dir=tmp_path)
    assert saved_again == 0


@patch("gex_core.uw_loader.fetch_uw_spot_exposures_intraday", return_value=pd.DataFrame())
def test_backfill_intraday_minutes_skips_market_holiday(mock_intraday, tmp_path):
    saved = backfill_intraday_minutes("SPX", "2026-04-03", export_dir=tmp_path)
    assert saved == 0
    mock_intraday.assert_called_once()
