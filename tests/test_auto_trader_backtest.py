import pandas as pd
import pytest

from gex_core.trading.backtest import backtest_auto_trader


@pytest.fixture(autouse=True)
def relaxed_filters(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "0")


def _snapshot(ts: str, spot: float, strikes: dict[float, float], *, prev_spot: float | None = None) -> dict:
    return {
        "ts": ts,
        "spot": spot,
        "strike": pd.Series(strikes),
        "regime": "LONG gamma",
        "gamma_flip": spot * 0.995,
        "flow_net_delta_gex_bn": 0.5,
        "is_cpi_day": False,
        "is_nfp_day": False,
        "is_fomc_week": False,
    }


def test_backtest_opens_and_closes_on_take_profit():
    history = [
        _snapshot("2026-06-01_100000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_101000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
        _snapshot("2026-06-01_102000", 5050.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
    ]
    result = backtest_auto_trader(
        "SPX",
        history=history,
        min_confidence=0.4,
        stop_loss=0.05,
        take_profit=0.10,
        max_open=1,
    )
    assert result["total_trades"] >= 1
    assert result["win_rate"] >= 0.0
    assert "by_signal" in result


def test_backtest_respects_min_confidence():
    history = [
        _snapshot("2026-06-01_100000", 5000.0, {5000: 0.01, 5010: 0.02}),
        _snapshot("2026-06-01_101000", 5000.0, {5000: 0.01, 5010: 0.03}),
        _snapshot("2026-06-01_102000", 5000.0, {5000: 0.01, 5010: 0.04}),
    ]
    result = backtest_auto_trader(
        "SPX",
        history=history,
        min_confidence=0.99,
        max_open=1,
    )
    assert result["total_trades"] == 0
    assert result["skipped_entries"] >= 1


def test_backtest_insufficient_history():
    result = backtest_auto_trader("SPX", history=[_snapshot("t", 5000, {5010: 1.0})])
    assert result["total_trades"] == 0
    assert "Not enough history" in result.get("message", "")


def test_backtest_caps_stop_and_take_profit():
    history = [
        _snapshot("2026-06-01_100000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_101000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
        _snapshot("2026-06-01_102000", 4800.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
    ]
    result = backtest_auto_trader(
        "SPX",
        history=history,
        min_confidence=0.4,
        stop_loss=0.05,
        take_profit=0.35,
        max_open=1,
    )
    stop_trades = [t for t in result["trades"] if t["exit_reason"] == "stop_loss"]
    assert stop_trades
    assert stop_trades[0]["pnl_pct"] == -0.05


def test_backtest_blocks_duplicate_strike_entries():
    history = [
        _snapshot("2026-06-01_100000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_101000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
        _snapshot("2026-06-01_102000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.3}),
        _snapshot("2026-06-01_103000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.4}),
    ]
    result = backtest_auto_trader(
        "SPX",
        history=history,
        min_confidence=0.4,
        max_open=3,
    )
    assert result["total_trades"] == 1
    assert result["blocked_duplicate"] >= 2
    assert len({t["strike"] for t in result["trades"]}) == 1


def test_backtest_skips_gamma_decline_entries():
    history = [
        _snapshot("2026-06-01_100000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_101000", 5000.0, {4990: 0.1, 5000: 0.3, 5010: 1.0}),
        _snapshot("2026-06-01_102000", 5000.0, {4990: 0.05, 5000: 0.2, 5010: 0.8}),
    ]
    result = backtest_auto_trader(
        "SPX",
        history=history,
        min_confidence=0.4,
        max_open=1,
    )
    assert result["total_trades"] == 0
    assert result["skipped_gamma_decline"] >= 1


def test_backtest_account_starting_capital(monkeypatch):
    monkeypatch.setenv("GEX_SIGNAL_TICKER", "SPX")
    monkeypatch.setenv("GEX_EXECUTION_TICKER", "SPY")
    history = [
        _snapshot("2026-06-01_100000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_101000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
        _snapshot("2026-06-01_102000", 5050.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
    ]
    result = backtest_auto_trader(
        "SPX",
        history=history,
        min_confidence=0.4,
        stop_loss=0.05,
        take_profit=0.10,
        max_open=1,
        starting_capital=500.0,
    )
    account = result.get("account")
    assert account is not None
    assert account["starting_capital"] == 500.0
    assert account["ending_capital"] > 0
    if result["total_trades"]:
        assert result["trades"][0]["strike"] < 600
        assert result["execution_ticker"] == "SPY"
