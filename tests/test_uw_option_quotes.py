from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from gex_core.uw_option_quotes import (
    UwOptionMarkProvider,
    _intraday_rows_to_frame,
    _mid_at_instant,
    _parse_option_symbol,
    contract_mid_from_row,
    option_pnl_pct_from_mids,
)


def test_contract_mid_from_row_prefers_nbbo_mid():
    row = {"nbbo_bid": "1.00", "nbbo_ask": "1.20", "last_price": "0.50"}
    assert contract_mid_from_row(row) == 1.1


def test_contract_mid_from_row_falls_back_to_last():
    row = {"nbbo_bid": "0", "nbbo_ask": "0", "last_price": "0.75"}
    assert contract_mid_from_row(row) == 0.75


def test_option_pnl_pct_from_mids():
    pnl = option_pnl_pct_from_mids(entry_mid=1.0, current_mid=1.2)
    assert abs(pnl - 0.2) < 1e-9


def test_parse_option_symbol():
    assert _parse_option_symbol("SPY260606C00500000") == (500.0, "call")
    assert _parse_option_symbol("SPY260606P00495000") == (495.0, "put")


def test_intraday_rows_to_frame_sorts_and_parses():
    rows = [
        {"start_time": "2026-06-06T15:31:00.000000Z", "close": "1.10"},
        {"start_time": "2026-06-06T15:30:00.000000Z", "close": "1.00"},
    ]
    frame = _intraday_rows_to_frame(rows)
    assert len(frame) == 2
    assert frame.iloc[0]["mid"] == 1.0
    assert frame.iloc[1]["mid"] == 1.1


def test_mid_at_instant_uses_last_bar_before_snapshot():
    frame = pd.DataFrame(
        [
            {"time": datetime(2026, 6, 6, 15, 30, tzinfo=timezone.utc), "mid": 1.0},
            {"time": datetime(2026, 6, 6, 15, 32, tzinfo=timezone.utc), "mid": 1.5},
        ]
    )
    instant = datetime(2026, 6, 6, 15, 31, tzinfo=timezone.utc)
    assert _mid_at_instant(frame, instant) == 1.0


@patch("gex_core.uw_option_quotes.fetch_uw_option_intraday")
def test_uw_mark_provider_mid_at(mock_intraday, monkeypatch, tmp_path):
    monkeypatch.setenv("GEX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GEX_EXECUTION_TICKER", "SPY")
    mock_intraday.return_value = [
        {"start_time": "2026-06-06T14:30:00.000000Z", "close": "2.00"},
        {"start_time": "2026-06-06T14:32:00.000000Z", "close": "2.40"},
    ]
    provider = UwOptionMarkProvider(api_key="test", use_disk_cache=False)
    mid = provider.mid_at(ts="2026-06-06_143100", strike=500.0, option_type="call")
    assert mid == 2.0
    assert provider.hits == 1
