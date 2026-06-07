import pandas as pd

from gex_core.trading.backtest_agent import (
    current_trader_parameters,
    format_backtest_reply,
    run_agent_backtest,
    summarize_backtest_for_ai,
    user_wants_backtest,
)


def test_user_wants_backtest_detects_common_phrases():
    assert user_wants_backtest("Can you backtest the current strategy?")
    assert user_wants_backtest("Walk-forward performance with these parameters")
    assert not user_wants_backtest("What is the gamma flip?")


def test_current_trader_parameters_includes_risk(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_RISK_PER_TRADE_PCT", "0.50")
    monkeypatch.setenv("GEX_TRADER_STOP_LOSS_PCT", "0.04")
    params = current_trader_parameters()
    assert params["risk_per_trade_pct"] == 0.50
    assert params["stop_loss_pct"] == 0.04


def test_run_agent_backtest_uses_current_parameters(monkeypatch):
    monkeypatch.setenv("GEX_TRADER_STRICT_FILTERS", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    monkeypatch.setenv("GEX_TRADER_RISK_PER_TRADE_PCT", "0.25")
    history = [
        {
            "ts": "2026-06-01_100000",
            "spot": 5000.0,
            "strike": pd.Series({5010: 2.0}),
            "regime": "LONG gamma",
            "gamma_flip": 4990.0,
            "flow_net_delta_gex_bn": 0.5,
            "is_cpi_day": False,
            "is_nfp_day": False,
            "is_fomc_week": False,
        },
        {
            "ts": "2026-06-01_101000",
            "spot": 5050.0,
            "strike": pd.Series({5010: 2.2}),
            "regime": "LONG gamma",
            "gamma_flip": 4990.0,
            "flow_net_delta_gex_bn": 0.5,
            "is_cpi_day": False,
            "is_nfp_day": False,
            "is_fomc_week": False,
        },
    ]

    from gex_core.trading import backtest as backtest_mod

    monkeypatch.setattr(backtest_mod, "_build_history_impl", lambda *a, **k: history)

    summary = run_agent_backtest("SPX", lookback_days=7, max_snapshots=50)
    assert summary["parameters"]["risk_per_trade_pct"] == 0.25
    assert "total_trades" in summary
    assert "window" in summary


def test_format_backtest_reply_includes_win_rate():
    text = format_backtest_reply(
        summarize_backtest_for_ai(
            {
                "parameters": current_trader_parameters(),
                "date_from": "2026-06-01",
                "date_to": "2026-06-02",
                "snapshots": 12,
                "total_trades": 3,
                "win_rate": 0.67,
                "total_pnl_usd": 120.0,
                "account": {"return_pct": 0.24},
            }
        )
    )
    assert "67%" in text or "win rate" in text
    assert "12 snapshots" in text


def test_api_agent_backtest_endpoint():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/api/agent/backtest?lookback_days=1&max_snapshots=20")
    assert response.status_code == 200
    payload = response.get_json()
    assert "parameters" in payload
    assert payload["parameters"]["risk_per_trade_pct"] > 0
