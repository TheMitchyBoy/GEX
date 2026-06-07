import pandas as pd

from gex_core.trading.strategy_viz import build_strategy_chart, build_strategy_dashboard, build_strategy_state


def test_build_strategy_state_with_signals(monkeypatch):
    monkeypatch.setattr("gex_core.trading.advisor._resolve_openai_config", lambda: None)
    monkeypatch.setenv("GEX_TRADER_CLEAR_FILTERS", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_MOMENTUM", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_FLIP_SIDE", "0")
    cur = pd.Series([0.2, 1.5, 0.4], index=[7380.0, 7390.0, 7400.0])
    prev = pd.Series([0.1, 0.5, 0.35], index=[7380.0, 7390.0, 7400.0])
    state = build_strategy_state(
        ticker="SPX",
        spot=7385.0,
        exposure=cur,
        previous_exposure=prev,
        snapshot={"regime": "LONG gamma", "gamma_flip": 7370.0},
        prev_spot=7380.0,
    )
    assert state["signals"]["available"]
    assert "rules" in state
    assert "filters" in state


def test_build_strategy_dashboard_returns_chart(monkeypatch):
    monkeypatch.setattr("gex_core.trading.advisor._resolve_openai_config", lambda: None)
    monkeypatch.setenv("GEX_TRADER_CLEAR_FILTERS", "0")
    monkeypatch.setenv("GEX_TRADER_MIN_GAMMA_DELTA", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_MOMENTUM", "0")
    monkeypatch.setenv("GEX_TRADER_REQUIRE_FLIP_SIDE", "0")
    cur = pd.Series([0.2, 1.5, 0.4], index=[7380.0, 7390.0, 7400.0])
    prev = pd.Series([0.1, 0.5, 0.35], index=[7380.0, 7390.0, 7400.0])
    out = build_strategy_dashboard(
        ticker="SPX",
        spot=7385.0,
        exposure=cur,
        previous_exposure=prev,
        snapshot={"regime": "LONG gamma"},
        prev_spot=7380.0,
    )
    assert "chart_json" in out
    assert "state" in out
    fig = build_strategy_chart(spot=7385.0, exposure=cur, state=out["state"])
    assert len(fig.data) >= 1
