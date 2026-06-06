from gex_core.trading.execution import build_webull_option_symbol, map_execution_strike, resolve_execution_spot


def test_map_execution_strike_ratio():
    assert map_execution_strike(5910.0, signal_spot=5910.0, execution_spot=591.0) == 591


def test_build_option_symbol_put():
    sym = build_webull_option_symbol(
        underlying="SPY",
        expire_date="2026-06-06",
        option_type="put",
        strike=585.0,
    )
    assert sym.endswith("P00585000")


def test_resolve_execution_spot_from_spx(monkeypatch):
    monkeypatch.setenv("GEX_EXECUTION_TICKER", "SPY")
    monkeypatch.setenv("GEX_SIGNAL_TICKER", "SPX")
    spot = resolve_execution_spot(signal_spot=5900.0)
    assert spot == 590.0
