from gex_core.trading.webull_broker import build_option_order, limit_price_for_buy


def test_build_option_order_payload():
    order = build_option_order(
        client_order_id="abc123",
        symbol="SPX",
        strike=5900.0,
        option_type="call",
        expire_date="2026-06-06",
        side="BUY",
        quantity=1,
        limit_price=12.5,
    )
    assert order["symbol"] == "SPX"
    assert order["side"] == "BUY"
    assert order["legs"][0]["strike_price"] == "5900.00"
    assert order["legs"][0]["option_type"] == "CALL"


def test_limit_price_for_buy_adds_buffer():
    px = limit_price_for_buy(5900.0, 5900.0, side="buy")
    assert px > 5900.0 * 0.001
