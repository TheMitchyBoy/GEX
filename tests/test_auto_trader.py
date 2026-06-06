import pandas as pd

from gex_core.trading.advisor import _rule_based_advice
from gex_core.trading.engine import run_trading_cycle
from gex_core.trading.journal import (
    get_performance_summary,
    is_trader_armed,
    list_open_trades,
    open_trade,
    set_trader_armed,
)
from gex_core.trading.paper_broker import estimate_option_pnl_pct
from gex_core.trading.signals import compute_gamma_signals


def test_compute_gamma_signals_picks_max_and_fastest():
    cur = pd.Series([0.2, 1.5, 0.4, 0.3], index=[7380.0, 7390.0, 7400.0, 7410.0])
    prev = pd.Series([0.1, 0.5, 0.35, 0.3], index=[7380.0, 7390.0, 7400.0, 7410.0])
    out = compute_gamma_signals(cur, prev, spot=7385.0)
    assert out["available"]
    assert out["max_positive_gamma"]["strike"] == 7390.0
    assert out["fastest_gamma_increase"]["gamma_delta"] == 1.0


def test_paper_broker_stop_loss_threshold():
    pnl = estimate_option_pnl_pct("call", entry_spot=5000, current_spot=4975, strike=5020)
    assert pnl < -0.05


def test_trading_cycle_opens_on_armed_trader(tmp_path, monkeypatch):
    db = tmp_path / "journal.db"
    monkeypatch.setenv("GEX_TRADING_DB", str(db))
    monkeypatch.setenv("GEX_AUTO_TRADER", "1")
    monkeypatch.setenv("GEX_TRADER_MIN_AI_CONFIDENCE", "0.4")

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


def test_rule_based_advice_downweights_poor_history():
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
    advice = _rule_based_advice(signals, memory)
    assert advice["confidence"] < 0.55


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
