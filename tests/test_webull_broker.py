from gex_core.trading.execution import (
    build_webull_option_symbol,
    map_execution_strike,
)
from gex_core.trading.webull_broker import (
    _order_avg_price,
    _order_filled_qty,
    _order_status,
    build_option_order,
    clear_webull_equity_cache,
    fetch_total_account_value,
    limit_price_for_buy,
    parse_total_account_value,
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
    strike = map_execution_strike(5900.0, signal_spot=5900.0, execution_spot=589.41)
    assert strike == 589


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


def test_parse_total_account_value_prefers_net_liquidation():
    balance = {
        "total_net_liquidation_value": "727687.04",
        "total_cash_balance": "485705.0",
        "account_currency_assets": [{"currency": "USD", "net_liquidation_value": "700000.0"}],
    }
    assert parse_total_account_value(balance) == 727687.04


def test_parse_total_account_value_falls_back_to_currency_assets():
    balance = {
        "account_currency_assets": [{"currency": "USD", "net_liquidation_value": "512.75"}],
    }
    assert parse_total_account_value(balance) == 512.75


def test_fetch_total_account_value_uses_cache(monkeypatch):
    clear_webull_equity_cache()
    calls = {"n": 0}

    def _fake_balance(*_a, **_k):
        calls["n"] += 1
        return {
            "code": 0,
            "data": {"total_net_liquidation_value": "1250.50"},
        }

    monkeypatch.setenv("GEX_WEBULL_ACCOUNT_ID", "acct-1")
    monkeypatch.setenv("GEX_WEBULL_APP_KEY", "key")
    monkeypatch.setenv("GEX_WEBULL_APP_SECRET", "secret")
    monkeypatch.setattr("gex_core.trading.webull_broker.fetch_account_balance", _fake_balance)

    first = fetch_total_account_value()
    second = fetch_total_account_value()
    assert first == 1250.50
    assert second == 1250.50
    assert calls["n"] == 1


def test_get_account_equity_uses_webull_live_value(monkeypatch):
    from gex_core.trading import journal

    monkeypatch.setenv("GEX_TRADER_PAPER", "0")
    monkeypatch.setenv("GEX_TRADER_LIVE_CONFIRM", "1")
    monkeypatch.setenv("GEX_WEBULL_ACCOUNT_ID", "acct-1")
    monkeypatch.setenv("GEX_WEBULL_APP_KEY", "key")
    monkeypatch.setenv("GEX_WEBULL_APP_SECRET", "secret")
    monkeypatch.setenv("GEX_TRADER_ACCOUNT_EQUITY", "500")
    monkeypatch.setattr(
        "gex_core.trading.webull_broker.fetch_total_account_value",
        lambda **kwargs: 2500.0,
    )

    assert journal.get_account_equity() == 2500.0
    assert journal.get_account_equity_source() == "webull_live"
