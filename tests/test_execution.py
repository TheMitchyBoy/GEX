import pytest

from gex_core.trading.execution import (
    build_webull_option_symbol,
    map_execution_strike,
    record_spot_ratio,
    resolve_execution_spot,
    spot_scale_ratio,
    sync_execution_context,
)


def test_spot_scale_ratio_real_market_example():
    """SPX 7383.74 / SPY 737.55 — not exactly 10:1."""
    ratio = spot_scale_ratio(signal_spot=7383.74, execution_spot=737.55)
    assert abs(ratio - 0.099888) < 0.0001
    assert abs(ratio - (1 / 10)) > 0.0001


def test_map_execution_strike_real_market_example():
    strike = map_execution_strike(7385.0, signal_spot=7383.74, execution_spot=737.55)
    # 7385 * (737.55 / 7383.74) ≈ 737.67 → 738
    assert strike == 738


def test_map_execution_strike_spx_magnet_at_spot():
    strike = map_execution_strike(7383.74, signal_spot=7383.74, execution_spot=737.55)
    assert strike == 738


def test_map_execution_strike_requires_both_spots():
    with pytest.raises(ValueError):
        map_execution_strike(7385.0, signal_spot=0, execution_spot=737.55)


def test_build_option_symbol_put():
    sym = build_webull_option_symbol(
        underlying="SPY",
        expire_date="2026-06-06",
        option_type="put",
        strike=737.55,
    )
    assert sym.startswith("SPY")
    assert "P00737550" in sym or sym.endswith("P00737550")


def test_resolve_execution_spot_uses_cached_ratio(monkeypatch):
    monkeypatch.setenv("GEX_EXECUTION_TICKER", "SPY")
    monkeypatch.setenv("GEX_SIGNAL_TICKER", "SPX")
    record_spot_ratio(signal_spot=7383.74, execution_spot=737.55)

    monkeypatch.setattr(
        "gex_core.trading.execution._fetch_live_spot",
        lambda _symbol: None,
    )
    spot = resolve_execution_spot(signal_spot=7383.74)
    assert spot is not None
    assert abs(spot - 737.55) < 0.05


def test_resolve_execution_spot_never_divides_by_ten(monkeypatch):
    monkeypatch.setenv("GEX_EXECUTION_TICKER", "SPY")
    monkeypatch.setenv("GEX_SIGNAL_TICKER", "SPX")
    monkeypatch.setattr("gex_core.trading.execution._cached_ratio", lambda: None)
    monkeypatch.setattr("gex_core.trading.execution._fetch_live_spot", lambda _symbol: None)

    assert resolve_execution_spot(signal_spot=7383.74) is None


def test_sync_execution_context_includes_ratio():
    ctx = sync_execution_context(signal_spot=7383.74)
    assert ctx["signal_spot"] == 7383.74
    assert ctx["execution_ticker"] == "SPY"
