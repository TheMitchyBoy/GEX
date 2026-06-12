import pandas as pd

from gex_core.trading.strategy_viz import (
    _chart_exposure_window,
    _gamma_bar_marker_styles,
    _gamma_change_points,
    build_strategy_chart,
    build_strategy_dashboard,
    build_strategy_state,
    build_wall_strategy_dashboard,
    build_wall_strategy_state,
)


def test_chart_exposure_window_shows_positive_gamma_below_spot():
    spot = 7580.0
    strikes = list(range(7420, 7621, 5))
    values = [1.0 if s < spot and s in {7430, 7475, 7525, 7570, 7575} else -0.5 for s in strikes]
    series = pd.Series(values, index=strikes, dtype=float)
    window = _chart_exposure_window(series, spot)
    below = window[window.index < spot]
    assert (below > 0).sum() >= 1


def test_chart_exposure_window_keeps_dense_atm_strikes():
    series = pd.Series(
        {7000 + i * 5: float(i % 3 - 1) for i in range(200)},
    )
    window = _chart_exposure_window(series, 7383.0, window_pct=0.025, max_strikes=65)
    assert len(window) >= 10
    steps = sorted(window.index.astype(float))
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    assert max(gaps) <= 10.0
    assert window.index.min() <= 7383.0 <= window.index.max()


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


def test_chart_exposure_window_one_percent_band():
    spot = 6000.0
    series = pd.Series(
        {float(s): (1.0 if s == 6000 else -0.5) for s in range(5880, 6121, 5)},
    )
    window = _chart_exposure_window(series, spot, window_pct=0.01, max_strikes=40)
    assert window.index.min() >= spot * 0.99 - 0.01
    assert window.index.max() <= spot * 1.01 + 0.01
    assert 6000.0 in window.index.astype(float).tolist()


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


def test_build_strategy_dashboard_near_spot_window(monkeypatch):
    monkeypatch.setattr("gex_core.trading.advisor._resolve_openai_config", lambda: None)
    monkeypatch.setenv("GEX_TRADER_CLEAR_FILTERS", "0")
    cur = pd.Series([0.2, 1.5, 0.4, -0.2], index=[7300.0, 7380.0, 7390.0, 7500.0])
    prev = pd.Series([0.1, 0.5, 0.35, -0.1], index=[7300.0, 7380.0, 7390.0, 7500.0])
    out = build_strategy_dashboard(
        ticker="SPX",
        spot=7385.0,
        exposure=cur,
        previous_exposure=prev,
        snapshot={"regime": "LONG gamma"},
        window_pct=0.01,
        max_strikes=40,
    )
    assert out["window_pct"] == 0.01


def test_build_wall_strategy_state_finds_low_and_high_walls(monkeypatch):
    monkeypatch.setenv("GEX_WALL_SIGNAL_FILTERS", "0")
    exposure = pd.Series(
        [-2.0, -0.5, 0.3, 2.5, 1.0],
        index=[7305.0, 7350.0, 7380.0, 7390.0, 7420.0],
    )
    state = build_wall_strategy_state(
        ticker="SPX",
        spot=7385.0,
        exposure=exposure,
        snapshot={"regime": "LONG gamma"},
        window_pct=0.01,
    )
    assert state["strategy_mode"] == "wall"
    assert state["signals"]["available"]
    assert state["signals"]["recommended"]["signal_type"] == "min_gamma_strike"
    assert state["signals"]["max_gamma_strike"]["signal_type"] == "max_gamma_strike"


def test_gamma_bar_marker_styles_heats_large_delta():
    window = pd.Series([1.0, -0.5, 0.2], index=[7380.0, 7390.0, 7400.0])
    prior = pd.Series([0.2, -0.5, 0.2], index=[7380.0, 7390.0, 7400.0])
    _colors, opacities = _gamma_bar_marker_styles(window, prior)
    assert max(opacities) > min(opacities)


def test_gamma_change_points_detects_largest_moves():
    current = pd.Series([1.0, -0.5, 0.2], index=[7380.0, 7390.0, 7400.0])
    trail = [
        {
            "label": "10:20",
            "series": pd.Series([0.4, -0.5, 0.2], index=[7380.0, 7390.0, 7400.0]),
            "age": 1,
        }
    ]
    points = _gamma_change_points(current, trail, window_index=current.index, top_n=3)
    assert points
    top = max(points, key=lambda p: abs(p["delta"]))
    assert top["strike"] == 7380.0
    assert top["delta"] > 0.5


def test_build_strategy_chart_includes_gamma_history_traces(monkeypatch):
    monkeypatch.setattr("gex_core.trading.advisor._resolve_openai_config", lambda: None)
    monkeypatch.setenv("GEX_TRADER_CLEAR_FILTERS", "0")
    cur = pd.Series([0.2, 1.5, 0.4], index=[7380.0, 7390.0, 7400.0])
    prev = pd.Series([0.1, 0.5, 0.35], index=[7380.0, 7390.0, 7400.0])
    state = build_strategy_state(
        ticker="SPX",
        spot=7385.0,
        exposure=cur,
        previous_exposure=prev,
        snapshot={"regime": "LONG gamma"},
        prev_spot=7380.0,
    )
    trail = [{"label": "prior", "series": prev, "age": 1, "spot": 7380.0}]
    fig = build_strategy_chart(
        spot=7385.0,
        exposure=cur,
        state=state,
        exposure_trail=trail,
        previous_exposure=prev,
    )
    scatter_traces = [t for t in fig.data if getattr(t, "mode", "") and "markers" in t.mode]
    assert len(scatter_traces) >= 2
    layers = {t.meta.get("layer") for t in fig.data if getattr(t, "meta", None)}
    assert "gamma_bars" in layers
    assert "gamma_history" in layers


def test_build_strategy_chart_includes_signal_path_trace(monkeypatch):
    monkeypatch.setattr("gex_core.trading.advisor._resolve_openai_config", lambda: None)
    monkeypatch.setenv("GEX_TRADER_CLEAR_FILTERS", "0")
    cur = pd.Series([-2.0, 0.5, 1.0], index=[7375.0, 7385.0, 7395.0])
    trail_series = [
        {"label": "t-3", "series": pd.Series([-1.5, 0.4, 0.8], index=[7375.0, 7385.0, 7395.0]), "age": 3, "spot": 7385.0},
        {"label": "t-2", "series": pd.Series([-1.8, 0.45, 0.9], index=[7375.0, 7385.0, 7395.0]), "age": 2, "spot": 7385.0},
        {"label": "t-1", "series": pd.Series([-1.9, 0.48, 0.95], index=[7375.0, 7385.0, 7395.0]), "age": 1, "spot": 7385.0},
    ]
    state = build_strategy_state(
        ticker="SPX",
        spot=7385.0,
        exposure=cur,
        previous_exposure=trail_series[-1]["series"],
        snapshot={"regime": "LONG gamma"},
    )
    fig = build_strategy_chart(
        spot=7385.0,
        exposure=cur,
        state=state,
        exposure_trail=trail_series,
        wall_mode=True,
        window_pct=0.01,
    )
    encoded = fig.to_json()
    assert "Signal path" in encoded or "signal_trail" in encoded


def test_build_wall_strategy_dashboard_returns_chart(monkeypatch):
    monkeypatch.setenv("GEX_WALL_SIGNAL_FILTERS", "0")
    exposure = pd.Series(
        [-2.0, -0.5, 0.3, 2.5],
        index=[7350.0, 7375.0, 7385.0, 7395.0],
    )
    out = build_wall_strategy_dashboard(
        ticker="SPX",
        spot=7385.0,
        exposure=exposure,
        snapshot={"regime": "LONG gamma"},
        window_pct=0.01,
        max_strikes=40,
    )
    assert out["strategy_mode"] == "wall"
    assert "chart_json" in out
    assert "Low wall" in out["chart_json"] or "min_gamma_strike" in out["chart_json"]
