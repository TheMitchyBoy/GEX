import pandas as pd

from gex_core.trading.engine import _apply_exit, _check_exits
from gex_core.trading.journal import list_open_trades, open_trade, set_trader_armed


def test_partial_exit_keeps_remaining_qty(tmp_path, monkeypatch):
    db = tmp_path / "journal.db"
    monkeypatch.setenv("GEX_TRADING_DB", str(db))
    monkeypatch.setenv("GEX_EXECUTION_TICKER", "SPY")
    monkeypatch.setenv("GEX_SIGNAL_TICKER", "SPX")
    monkeypatch.setenv("GEX_TRADER_PAPER", "1")

    tid = open_trade(
        ticker="SPX",
        option_type="call",
        strike=590.0,
        entry_spot=590.0,
        entry_premium=3.0,
        signal_type="max_positive_gamma",
        signal_strike=5900.0,
        signal_gamma=1.5,
        gamma_delta=0.2,
        ai_confidence=0.7,
        ai_reason="test",
        qty=2.0,
        meta={"partial_taken": False, "exit_profile": {"partial_take_profit": 0.15, "hold_for_target": False}},
    )

    class FakeBroker:
        name = "paper"

        def position_pnl_pct(self, trade, *, spot):
            return 0.20

        def sell_option(self, **kwargs):
            return {"ok": True, "filled_premium": 3.5}

    monkeypatch.setattr("gex_core.trading.engine.get_broker", lambda: FakeBroker())
    monkeypatch.setattr(
        "gex_core.trading.engine.evaluate_exit",
        lambda *a, **k: ("take_profit_partial", 0.18),
    )

    exits = _check_exits("SPX", spot=5900.0)
    assert exits
    assert exits[0].get("partial") is True
    open_rows = list_open_trades("SPX")
    assert len(open_rows) == 1
    assert float(open_rows[0]["qty"]) == 1.0


def test_apply_exit_full_close(tmp_path, monkeypatch):
    db = tmp_path / "journal.db"
    monkeypatch.setenv("GEX_TRADING_DB", str(db))
    tid = open_trade(
        ticker="SPX",
        option_type="call",
        strike=590.0,
        entry_spot=590.0,
        entry_premium=3.0,
        signal_type="test",
        signal_strike=5900.0,
        signal_gamma=1.0,
        gamma_delta=0.1,
        ai_confidence=0.6,
        ai_reason="test",
        qty=1.0,
        meta={"underlying": "SPY", "expire_date": "2026-06-06"},
    )
    pos = list_open_trades("SPX")[0]

    class FakeBroker:
        name = "paper"

        def sell_option(self, **kwargs):
            return {"ok": True}

    result = _apply_exit(
        ticker="SPX",
        pos=pos,
        meta=pos.get("meta") or {},
        spot=591.0,
        exit_reason="stop_loss",
        exit_pnl=-0.05,
        sell_qty=1,
        broker=FakeBroker(),
    )
    assert result
    assert not result.get("partial")
    assert not list_open_trades("SPX")
