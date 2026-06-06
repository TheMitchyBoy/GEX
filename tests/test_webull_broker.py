from gex_core.trading.execution import (
    build_webull_option_symbol,
    map_execution_strike,
)
from gex_core.trading.webull_broker import (
    _order_avg_price,
    _order_filled_qty,
    _order_status,
    build_option_order,
    limit_price_for_buy,
)


def test_build_webull_option_symbol_spy():
    sym = build_webull_option_symbol(
        underlying="SPY",
        expire_date="2026-06-06",
        option_type="call",
        strike=590.0,
    )
    assert sym == "SPY260606C00590000"


def test_map_execution_strike_spx_to_spy():
    strike = map_execution_strike(5900.0, signal_spot=5900.0, execution_spot=590.0)
    assert strike == 590


def test_build_option_order_payload():
    order = build_option_order(
        client_order_id="abc123",
        symbol="SPY",
        strike=590.0,
        option_type="call",
        expire_date="2026-06-06",
        side="BUY",
        quantity=1,
        limit_price=3.5,
    )
    assert order["symbol"] == "SPY"
    assert order["side"] == "BUY"
    assert order["legs"][0]["strike_price"] == "590.00"
    assert order["legs"][0]["option_type"] == "CALL"


def test_limit_price_for_buy_fallback():
    px = limit_price_for_buy(590.0, 590.0, side="buy")
    assert px > 0


def test_order_fill_parsing():
    order = {"status": "FILLED", "filled_quantity": 2, "avg_fill_price": 3.25}
    assert _order_status(order) == "FILLED"
    assert _order_filled_qty(order) == 2
    assert _order_avg_price(order) == 3.25
