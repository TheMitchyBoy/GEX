"""Weekend exclusion for auto-trader backtests and entry windows."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from gex_core.market_time import (
    export_ts_entry_window_ok,
    export_ts_is_trading_day,
    filter_trading_history,
    is_trading_weekday,
)
from gex_core.trading.backtest import backtest_auto_trader


def test_is_trading_weekday_rejects_saturday():
    sat = datetime(2026, 6, 6, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    assert not is_trading_weekday(now=sat)


def test_is_trading_weekday_accepts_friday():
    fri = datetime(2026, 6, 5, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    assert is_trading_weekday(now=fri)


def test_export_ts_is_trading_day_saturday_snapshot():
    # 14:09 UTC on June 6 2026 is Saturday morning ET.
    assert not export_ts_is_trading_day("2026-06-06_140933")


def test_export_ts_entry_window_ok_blocks_weekend_even_without_time_filter(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_ENTRY_TIME_FILTER", "0")
    assert not export_ts_entry_window_ok("2026-06-06_140933")
    assert not export_ts_entry_window_ok("2026-06-07_031659")


def test_filter_trading_history_drops_weekends():
    history = [
        {"ts": "2026-06-05_143000", "spot": 5000.0},
        {"ts": "2026-06-06_140933", "spot": 5000.0},
        {"ts": "2026-06-07_031659", "spot": 5000.0},
        {"ts": "2026-06-08_143000", "spot": 5000.0},
    ]
    filtered = filter_trading_history(history, session_only=True)
    assert [row["ts"] for row in filtered] == ["2026-06-05_143000", "2026-06-08_143000"]


def _snapshot(ts: str, spot: float, strikes: dict[float, float]) -> dict:
    return {
        "ts": ts,
        "spot": spot,
        "strike": pd.Series(strikes),
        "regime": "LONG gamma",
        "gamma_flip": spot * 0.995,
        "flow_net_delta_gex_bn": 0.5,
        "interval_minutes": 10,
    }


def test_backtest_excludes_weekend_snapshots(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "0")
    monkeypatch.setenv("GEX_TRADER_ENTRY_TIME_FILTER", "0")
    monkeypatch.setenv("GEX_TRADER_SESSION_ONLY", "1")
    history = [
        _snapshot("2026-06-05_143000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-05_144000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
        _snapshot("2026-06-06_140933", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.3}),
        _snapshot("2026-06-06_142933", 5050.0, {4990: 0.2, 5000: 0.5, 5010: 2.3}),
    ]
    result = backtest_auto_trader("SPX", history=history, stop_loss=0.05, take_profit=0.10)
    assert result["weekend_snapshots_excluded"] == 2
    assert result["snapshots"] == 2
    for trade in result.get("trades") or []:
        assert "2026-06-06" not in trade["entry_ts"]
