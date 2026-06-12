import pandas as pd
import pytest

from gex_core.trading.advisor import _rule_based_advice
from gex_core.trading.engine import run_trading_cycle
from gex_core.trading.filters import MarketContext, evaluate_entry_filters
from gex_core.trading.journal import (
    get_performance_summary,
    is_trader_armed,
    list_open_trades,
    open_trade,
    set_trader_armed,
)
from gex_core.trading.paper_broker import estimate_option_pnl_pct
from gex_core.trading.signals import compute_gamma_signals


def test_compute_gamma_signals_picks_max_and_fastest(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MAX_GAMMA_ONLY", "0")
    cur = pd.Series([0.2, 1.5, 0.4, 0.3], index=[7380.0, 7390.0, 7400.0, 7410.0])
    prev = pd.Series([0.1, 0.5, 0.35, 0.3], index=[7380.0, 7390.0, 7400.0, 7410.0])
    out = compute_gamma_signals(cur, prev, spot=7385.0)
    assert out["available"]
    assert out["max_positive_gamma"]["gamma_bn"] == 1.5
    assert out["fastest_gamma_increase"]["gamma_delta"] == 1.0
    assert out["recommended"]["signal_type"] == "fastest_gamma_increase"
    assert out["recommended"]["strike"] in {7380.0, 7390.0}


def test_compute_gamma_signals_switches_when_max_gamma_declines(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MAX_GAMMA_ONLY", "0")
    cur = pd.Series([0.2, 1.0, 0.8, 0.3], index=[7380.0, 7390.0, 7385.0, 7410.0])
    prev = pd.Series([0.2, 1.5, 0.4, 0.3], index=[7380.0, 7390.0, 7385.0, 7410.0])
    out = compute_gamma_signals(cur, prev, spot=7387.0)
    assert out["available"]
    # Direction locked to max-gamma magnet (7390 → call); fastest put at 7385 is excluded.
    assert out["recommended"]["signal_type"] == "max_positive_gamma"
    assert out["recommended"]["option_type"] == "call"
    assert out["master_direction"] == "call"


def test_compute_gamma_signals_allows_flat_max_gamma(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MAX_GAMMA_ONLY", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    cur = pd.Series([0.2, 1.5, 0.4, 0.3], index=[7380.0, 7390.0, 7388.0, 7410.0])
    prev = pd.Series([0.2, 1.5, 0.25, 0.3], index=[7380.0, 7390.0, 7388.0, 7410.0])
    out = compute_gamma_signals(cur, prev, spot=7387.0)
    assert out["available"]
    assert out["recommended"]["signal_type"] in {"max_positive_gamma", "fastest_gamma_increase"}
    assert out["max_pos_gamma_delta"] == 0.0


def test_compute_gamma_signals_blocks_when_max_gamma_declines():
    cur = pd.Series([0.1, 0.8, 0.3, 0.2], index=[7380.0, 7390.0, 7400.0, 7410.0])
    prev = pd.Series([0.2, 1.5, 0.5, 0.3], index=[7380.0, 7390.0, 7400.0, 7410.0])
    out = compute_gamma_signals(cur, prev, spot=7385.0)
    assert not out["available"]
    assert out["skip_reason"] == "gamma_declined"


def test_entry_filter_blocks_low_gamma_delta(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "1")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0.05")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_FLOW_ALIGN", "0")
    signals = {
        "available": True,
        "spot": 5000.0,
        "recommended": {
            "strike": 5010.0,
            "option_type": "call",
            "gamma_delta": 0.02,
            "score": 1.0,
        },
    }
    result = evaluate_entry_filters(signals, market=MarketContext(spot=5000.0, prev_spot=4990.0, regime="LONG gamma"))
    assert not result["approve"]
    assert result["filter"] == "gamma_delta"


def test_paper_broker_stop_loss_threshold():
    pnl = estimate_option_pnl_pct("call", entry_spot=5000, current_spot=4975, strike=5020)
    assert pnl < -0.05


def test_max_gamma_only_single_direction_candidate(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MAX_GAMMA_ONLY", "1")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    cur = pd.Series([0.2, 1.5, 0.4, 0.3], index=[7380.0, 7390.0, 7400.0, 7410.0])
    prev = pd.Series([0.1, 0.5, 0.35, 0.3], index=[7380.0, 7390.0, 7400.0, 7410.0])
    out = compute_gamma_signals(cur, prev, spot=7385.0)
    assert out["available"]
    assert out["recommended"]["signal_type"] == "max_positive_gamma"
    assert out["recommended"]["option_type"] == "call"
    assert len(out["candidates"]) == 1
    assert out["recommended"]["magnet_strike"] == 7390.0


def test_max_gamma_only_picks_lowest_negative_when_dominant(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MAX_GAMMA_ONLY", "1")
    monkeypatch.setenv("GEX_TRADER_TRADE_NEGATIVE_GAMMA", "1")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    monkeypatch.setenv("GEX_TRADER_MAX_STRIKE_DISTANCE_PCT", "0.05")
    cur = pd.Series([0.3, 0.5, -2.5, -0.4], index=[7370.0, 7380.0, 7390.0, 7400.0])
    prev = pd.Series([0.2, 0.4, -2.6, -0.3], index=[7370.0, 7380.0, 7390.0, 7400.0])
    out = compute_gamma_signals(cur, prev, spot=7385.0)
    assert out["available"]
    assert out["recommended"]["signal_type"] == "min_negative_gamma"
    assert out["recommended"]["magnet_strike"] == 7390.0
    assert out["recommended"]["option_type"] == "call"
    assert "min_negative_gamma" in out


def test_max_gamma_only_prefers_positive_when_negative_trading_off(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MAX_GAMMA_ONLY", "1")
    monkeypatch.setenv("GEX_TRADER_TRADE_NEGATIVE_GAMMA", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    cur = pd.Series([0.3, 0.5, -2.5, -0.4], index=[7370.0, 7380.0, 7390.0, 7400.0])
    prev = pd.Series([0.2, 0.4, -2.6, -0.3], index=[7370.0, 7380.0, 7390.0, 7400.0])
    out = compute_gamma_signals(cur, prev, spot=7385.0)
    assert out["available"]
    assert out["recommended"]["signal_type"] == "max_positive_gamma"
    assert out["recommended"]["magnet_strike"] == 7380.0


def test_max_gamma_only_blocks_declining_magnet(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MAX_GAMMA_ONLY", "1")
    cur = pd.Series([0.2, 1.0, 0.8, 0.3], index=[7380.0, 7390.0, 7385.0, 7410.0])
    prev = pd.Series([0.2, 1.5, 0.4, 0.3], index=[7380.0, 7390.0, 7385.0, 7410.0])
    out = compute_gamma_signals(cur, prev, spot=7387.0)
    assert not out["available"]
    assert out["skip_reason"] == "gamma_declined"


def test_signal_performance_weights_boost_winning_type(monkeypatch, tmp_path):
    monkeypatch.setenv("GEX_TRADING_DB", str(tmp_path / "journal.db"))
    from gex_core.trading.journal import open_trade, close_trade
    from gex_core.trading.signals import signal_performance_weights

    for _ in range(4):
        tid = open_trade(
            ticker="SPX",
            option_type="call",
            strike=7390.0,
            entry_spot=7385.0,
            entry_premium=10.0,
            signal_type="max_positive_gamma",
            signal_strike=7390.0,
            signal_gamma=1.5,
            gamma_delta=0.2,
            ai_confidence=0.7,
            ai_reason="test",
        )
        close_trade(tid, exit_spot=7395.0, exit_premium=12.0, pnl_pct=0.2, pnl_usd=200.0, exit_reason="take_profit")

    weights = signal_performance_weights("SPX")
    assert weights["max_positive_gamma"] > 1.0


def test_magnet_anchored_strike_uses_magnet(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_MAX_GAMMA_ONLY", "1")
    monkeypatch.setenv("GEX_TRADER_MAGNET_ANCHORED_STRIKES", "1")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    cur = pd.Series([0.2, 1.5, 0.4, 0.3], index=[7380.0, 7390.0, 7400.0, 7410.0])
    prev = pd.Series([0.1, 0.5, 0.35, 0.3], index=[7380.0, 7390.0, 7400.0, 7410.0])
    out = compute_gamma_signals(cur, prev, spot=7385.0)
    assert out["recommended"]["strike"] == 7390.0


def test_trading_cycle_opens_on_armed_trader(tmp_path, monkeypatch):
    db = tmp_path / "journal.db"
    monkeypatch.setenv("GEX_TRADING_DB", str(db))
    monkeypatch.setenv("GEX_AUTO_TRADER", "1")
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_MOMENTUM", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_FLIP_SIDE", "0")
    monkeypatch.setenv("GEX_TRADER_RISK_SIZING", "0")
    monkeypatch.setattr("gex_core.trading.advisor._resolve_openai_config", lambda: False)

    set_trader_armed(True)
    assert is_trader_armed()

    cur = pd.Series([0.3, 2.0, 0.5], index=[7380.0, 7390.0, 7400.0])
    prev = pd.Series([0.2, 0.4, 0.5], index=[7380.0, 7390.0, 7400.0])
    result = run_trading_cycle(
        ticker="SPX",
        spot=7385.0,
        exposure=cur,
        previous_exposure=prev,
        force=True,
    )
    assert result["ran"]
    assert result.get("entry") is not None
    assert len(list_open_trades("SPX")) >= 1


def test_rule_based_advice_downweights_poor_history(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "0")
    signals = {
        "recommended": {
            "signal_type": "max_positive_gamma",
            "strike": 7390,
            "gamma_bn": 1.2,
            "gamma_delta": 0.1,
            "score": 1.2,
            "option_type": "call",
            "rationale": "test",
        }
    }
    memory = {
        "performance": {
            "lessons": [],
            "by_signal": {
                "max_positive_gamma": {"count": 8, "win_rate": 0.2, "avg_pnl_pct": -0.08},
            },
        }
    }
    advice = _rule_based_advice(signals, memory, market=MarketContext(spot=7385.0, prev_spot=7380.0))
    assert advice["confidence"] < 0.65


def test_performance_summary_after_close(tmp_path, monkeypatch):
    db = tmp_path / "journal.db"
    monkeypatch.setenv("GEX_TRADING_DB", str(db))
    tid = open_trade(
        ticker="SPX",
        option_type="call",
        strike=7390,
        entry_spot=7385,
        entry_premium=10,
        signal_type="max_positive_gamma",
        signal_strike=7390,
        signal_gamma=1.5,
        gamma_delta=0.2,
        ai_confidence=0.7,
        ai_reason="test",
    )
    from gex_core.trading.journal import close_trade

    close_trade(tid, exit_spot=7400, exit_premium=12, pnl_pct=0.2, pnl_usd=200, exit_reason="take_profit")
    perf = get_performance_summary("SPX")
    assert perf["total_trades"] == 1
    assert perf["win_rate"] == 1.0
