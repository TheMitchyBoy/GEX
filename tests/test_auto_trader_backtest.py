import pandas as pd
import pytest

from gex_core.trading.backtest import backtest_auto_trader


@pytest.fixture(autouse=True)
def relaxed_filters(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_MAGNET_PROGRESS", "0")
    monkeypatch.setenv("GEX_TRADER_PREFER_SIGNAL", "")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_MOMENTUM", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_FLIP_SIDE", "0")


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
        stop_loss=0.05,
        take_profit=0.10,
        max_open=1,
    )
    assert result["total_trades"] >= 1
    assert result["win_rate"] >= 0.0
    assert "by_signal" in result


def test_backtest_tries_multiple_strike_candidates(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MAX_GAMMA_ONLY", "0")
    monkeypatch.setenv("GEX_TRADER_MAGNET_ANCHORED_STRIKES", "1")
    monkeypatch.setenv("GEX_TRADER_MULTI_STRIKE", "3")
    monkeypatch.setenv("GEX_TRADER_MAX_ENTRIES_PER_CYCLE", "3")
    history = [
        _snapshot("2026-06-01_100000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0, 5020: 1.8}),
        _snapshot("2026-06-01_101000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.2, 5020: 1.9}),
        _snapshot("2026-06-01_102000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.3, 5020: 2.0}),
        _snapshot("2026-06-01_103000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.4, 5020: 2.1}),
    ]
    result = backtest_auto_trader(
        "SPX",
        history=history,
        max_open=3,
    )
    assert result["total_trades"] >= 2
    assert len({t["strike"] for t in result["trades"]}) >= 2


def test_backtest_insufficient_history():
    result = backtest_auto_trader("SPX", history=[_snapshot("t", 5000, {5010: 1.0})])
    assert result["total_trades"] == 0
    assert "Not enough history" in result.get("message", "")


def test_backtest_caps_stop_and_take_profit(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_STOP_LOSS_PCT", "0.05")
    monkeypatch.setenv("GEX_TRADER_TAKE_PROFIT_PCT", "0.35")
    history = [
        _snapshot("2026-06-01_100000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_101000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
        _snapshot("2026-06-01_102000", 4800.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
    ]
    result = backtest_auto_trader(
        "SPX",
        history=history,
        stop_loss=0.05,
        take_profit=0.35,
        max_open=1,
    )
    stop_trades = [t for t in result["trades"] if t["exit_reason"] == "stop_loss"]
    assert stop_trades
    assert stop_trades[0]["pnl_pct"] == -0.05


def test_backtest_blocks_duplicate_strike_entries():
    history = [
        _snapshot("2026-06-01_100000", 5000.0, {5010: 2.0}),
        _snapshot("2026-06-01_101000", 5005.0, {5010: 2.2}),
        _snapshot("2026-06-01_102000", 5005.0, {5010: 2.3}),
        _snapshot("2026-06-01_103000", 5005.0, {5010: 2.4}),
    ]
    result = backtest_auto_trader(
        "SPX",
        history=history,
        max_open=3,
    )
    assert result["total_trades"] == 1
    assert result["blocked_duplicate"] >= 2
    assert len({t["strike"] for t in result["trades"]}) == 1


def test_backtest_enters_on_flat_gamma_magnet():
    history = [
        _snapshot("2026-06-01_100000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_101000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_102000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
    ]
    result = backtest_auto_trader(
        "SPX",
        history=history,
        max_open=1,
    )
    assert result["total_trades"] >= 1
    assert result["skipped_gamma_decline"] == 0


def test_backtest_skips_exits_across_snapshot_gaps():
    history = [
        _snapshot("2026-06-01_093000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_094000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
        _snapshot("2026-06-03_093000", 4800.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
    ]
    for row in history:
        row["interval_minutes"] = 10
    result = backtest_auto_trader(
        "SPX",
        history=history,
        stop_loss=0.05,
        take_profit=0.35,
        max_open=1,
    )
    assert result["total_trades"] == 1
    assert result["trades"][0]["exit_reason"] in {"session_gap", "time_stop", "backtest_end"}


def test_backtest_account_cash_matches_trade_pnl(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "0")
    monkeypatch.setenv("GEX_TRADER_RISK_SIZING", "0")
    history = [
        _snapshot("2026-06-01_100000", 5000.0, {4990: 0.2, 5000: 0.4, 5010: 2.0}),
        _snapshot("2026-06-01_101000", 5005.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
        _snapshot("2026-06-01_102000", 5050.0, {4990: 0.2, 5000: 0.5, 5010: 2.2}),
    ]
    result = backtest_auto_trader(
        "SPX",
        history=history,
        stop_loss=0.05,
        take_profit=0.10,
        max_open=1,
        starting_capital=500.0,
    )
    account = result.get("account")
    assert account is not None
    trade_pnl = sum(t["pnl_usd"] for t in result["trades"])
    expected_cash = 500.0 + trade_pnl
    assert abs(account["ending_capital"] - expected_cash) < 0.05


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
