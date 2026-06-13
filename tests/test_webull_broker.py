import os
from pathlib import Path

from gex_core.trading.config import webull_configured, webull_data_endpoint, webull_trade_endpoint
from gex_core.trading.execution import (
    build_webull_option_symbol,
    map_execution_strike,
)
from gex_core.trading.webull_broker import (
    migrate_legacy_webull_token,
    _order_avg_price,
    _order_filled_qty,
    _order_status,
    build_option_order,
    clear_webull_equity_cache,
    clear_webull_error_state,
    clear_webull_quote_cache,
    fetch_option_quote,
    fetch_total_account_value,
    limit_price_for_buy,
    note_webull_error,
    parse_total_account_value,
    reconnect_webull_auth,
    reset_webull_clients,
    webull_auth_status,
    webull_api_paused,
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


def test_webull_configured_respects_enabled_flag(monkeypatch):
    monkeypatch.setenv("GEX_WEBULL_APP_KEY", "key")
    monkeypatch.setenv("GEX_WEBULL_APP_SECRET", "secret")
    monkeypatch.setenv("GEX_WEBULL_ACCOUNT_ID", "acct-1")
    monkeypatch.delenv("GEX_WEBULL_ENABLED", raising=False)
    assert webull_configured() is False
    monkeypatch.setenv("GEX_WEBULL_ENABLED", "1")
    assert webull_configured() is True


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


def test_webull_trade_endpoint_defaults_to_production(monkeypatch):
    monkeypatch.delenv("GEX_WEBULL_ENDPOINT", raising=False)
    monkeypatch.delenv("GEX_WEBULL_USE_UAT", raising=False)
    assert webull_trade_endpoint() == "api.webull.com"
    assert webull_data_endpoint() == "broker-api.webull.com"


def test_webull_trade_endpoint_uat(monkeypatch):
    monkeypatch.setenv("GEX_WEBULL_USE_UAT", "1")
    monkeypatch.delenv("GEX_WEBULL_ENDPOINT", raising=False)
    assert webull_trade_endpoint() == "us-openapi-alb.uat.webullbroker.com"
    assert webull_data_endpoint() == "us-broker-api.uat.webullbroker.com"


def test_webull_auth_status_invalid_token_banner(tmp_path, monkeypatch):
    clear_webull_error_state()
    reset_webull_clients()
    monkeypatch.setenv("GEX_TRADER_PAPER", "0")
    monkeypatch.setenv("GEX_WEBULL_APP_KEY", "key")
    monkeypatch.setenv("GEX_WEBULL_APP_SECRET", "secret")
    monkeypatch.setenv("GEX_WEBULL_ACCOUNT_ID", "acct-1")
    monkeypatch.setenv("WEBULL_OPENAPI_TOKEN_DIR", str(tmp_path))
    token_file = tmp_path / "token.txt"
    token_file.write_text("abc123\n1700000000\nNORMAL\n", encoding="utf-8")

    note_webull_error(
        'HTTP Status: 401, Code: INVALID_TOKEN, Msg: 401 UNAUTHORIZED "permission denied", RequestID: x'
    )

    assert not token_file.exists()
    auth = webull_auth_status()
    assert auth["invalid_token"] is True
    assert auth["show_banner"] is True
    assert "401" in auth["headline"]


def test_webull_auth_status_invalid_token_pauses_api(monkeypatch):
    clear_webull_error_state()
    monkeypatch.setenv("GEX_TRADER_PAPER", "0")
    monkeypatch.setenv("GEX_WEBULL_APP_KEY", "key")
    monkeypatch.setenv("GEX_WEBULL_APP_SECRET", "secret")
    monkeypatch.setenv("GEX_WEBULL_ACCOUNT_ID", "acct-1")
    note_webull_error("HTTP Status: 401, Code: INVALID_TOKEN")

    auth = webull_auth_status()
    assert auth["invalid_token"] is True
    assert auth["pause_api"] is True
    assert webull_api_paused() is True


def test_reconnect_webull_auth_clears_error_state(tmp_path, monkeypatch):
    clear_webull_error_state()
    reset_webull_clients()
    monkeypatch.setenv("GEX_TRADER_PAPER", "0")
    monkeypatch.setenv("GEX_WEBULL_APP_KEY", "key")
    monkeypatch.setenv("GEX_WEBULL_APP_SECRET", "secret")
    monkeypatch.setenv("GEX_WEBULL_ACCOUNT_ID", "acct-1")
    monkeypatch.setenv("WEBULL_OPENAPI_TOKEN_DIR", str(tmp_path))
    token_file = tmp_path / "token.txt"
    token_file.write_text("abc123\n1700000000\nNORMAL\n", encoding="utf-8")
    note_webull_error("HTTP Status: 401, Code: INVALID_TOKEN")

    monkeypatch.setattr(
        "gex_core.trading.webull_broker.fetch_account_balance",
        lambda **kwargs: {"code": 0, "data": {"total_net_liquidation_value": "1000"}},
    )

    result = reconnect_webull_auth()
    assert result["ok"] is True
    assert webull_auth_status()["show_banner"] is False


def test_webull_auth_status_quote_subscription_banner(tmp_path, monkeypatch):
    clear_webull_error_state()
    reset_webull_clients()
    monkeypatch.setenv("GEX_TRADER_PAPER", "0")
    monkeypatch.setenv("GEX_WEBULL_APP_KEY", "key")
    monkeypatch.setenv("GEX_WEBULL_APP_SECRET", "secret")
    monkeypatch.setenv("GEX_WEBULL_ACCOUNT_ID", "acct-1")
    monkeypatch.setenv("WEBULL_OPENAPI_TOKEN_DIR", str(tmp_path))
    token_file = tmp_path / "token.txt"
    token_file.write_text("abc123\n1700000000\nNORMAL\n", encoding="utf-8")

    note_webull_error(
        "HTTP Status: 401, Code: Unauthorized, Msg: Insufficient permission, "
        "please subscribe to US_OPTION quotes., RequestID: x"
    )

    assert token_file.exists()
    auth = webull_auth_status()
    assert auth["quote_subscription_required"] is True
    assert auth["invalid_token"] is False
    assert auth["show_banner"] is True
    assert auth["can_reconnect"] is False
    assert "US options quotes not subscribed" in auth["headline"]


def test_webull_auth_status_rate_limit_banner(monkeypatch):
    clear_webull_error_state()
    monkeypatch.setenv("GEX_TRADER_PAPER", "0")
    monkeypatch.setenv("GEX_WEBULL_APP_KEY", "key")
    monkeypatch.setenv("GEX_WEBULL_APP_SECRET", "secret")
    monkeypatch.setenv("GEX_WEBULL_ACCOUNT_ID", "acct-1")
    note_webull_error("HTTP Status: 429, Code: TOO_MANY_REQUESTS")

    auth = webull_auth_status()
    assert auth["rate_limited"] is True
    assert auth["show_banner"] is True
    assert webull_api_paused() is True


def test_webull_auth_status_ok_without_sms(monkeypatch):
    clear_webull_error_state()
    monkeypatch.setenv("GEX_TRADER_PAPER", "0")
    monkeypatch.setenv("GEX_WEBULL_APP_KEY", "key")
    monkeypatch.setenv("GEX_WEBULL_APP_SECRET", "secret")
    monkeypatch.setenv("GEX_WEBULL_ACCOUNT_ID", "acct-1")

    auth = webull_auth_status()
    assert auth["show_banner"] is False
    assert auth["pause_api"] is False
    assert webull_api_paused() is False


def test_fetch_option_quote_uses_cache(monkeypatch):
    clear_webull_quote_cache()
    clear_webull_error_state()
    calls = {"n": 0}

    def _fake_snapshot(*_a, **_k):
        calls["n"] += 1
        return {"code": 0, "data": {"bid": 1.0, "ask": 1.2, "last": 1.1}}

    monkeypatch.setenv("GEX_WEBULL_APP_KEY", "key")
    monkeypatch.setenv("GEX_WEBULL_APP_SECRET", "secret")
    monkeypatch.setenv("GEX_WEBULL_ACCOUNT_ID", "acct-1")
    monkeypatch.setattr(
        "gex_core.trading.webull_broker._ensure_data_client",
        lambda: type(
            "D",
            (),
            {
                "option_market_data": type(
                    "M",
                    (),
                    {"get_option_snapshot": lambda *_a, **_k: _fake_snapshot()},
                )()
            },
        )(),
    )

    first = fetch_option_quote(
        underlying="SPY",
        option_type="call",
        strike=590.0,
        expire_date="2026-06-06",
    )
    second = fetch_option_quote(
        underlying="SPY",
        option_type="call",
        strike=590.0,
        expire_date="2026-06-06",
    )
    assert first["bid"] == 1.0
    assert second["bid"] == 1.0
    assert calls["n"] == 1


def test_fetch_option_quote_rate_limit_uses_stale_cache(monkeypatch):
    clear_webull_quote_cache()
    clear_webull_error_state()
    monkeypatch.setenv("GEX_TRADER_PAPER", "0")
    monkeypatch.setenv("GEX_WEBULL_APP_KEY", "key")
    monkeypatch.setenv("GEX_WEBULL_APP_SECRET", "secret")
    monkeypatch.setenv("GEX_WEBULL_ACCOUNT_ID", "acct-1")

    state = {"mode": "ok"}

    def _fake_snapshot(*_a, **_k):
        if state["mode"] == "ok":
            return {"code": 0, "data": {"bid": 2.0, "ask": 2.2, "last": 2.1}}
        return {"code": 429, "msg": "TOO_MANY_REQUESTS"}

    monkeypatch.setattr(
        "gex_core.trading.webull_broker._ensure_data_client",
        lambda: type(
            "D",
            (),
            {
                "option_market_data": type(
                    "M",
                    (),
                    {"get_option_snapshot": lambda *_a, **_k: _fake_snapshot()},
                )()
            },
        )(),
    )

    first = fetch_option_quote(
        underlying="SPY",
        option_type="call",
        strike=590.0,
        expire_date="2026-06-06",
        force_refresh=True,
    )
    state["mode"] = "rate_limited"
    second = fetch_option_quote(
        underlying="SPY",
        option_type="call",
        strike=590.0,
        expire_date="2026-06-06",
        force_refresh=True,
    )
    assert first["bid"] == 2.0
    assert second["bid"] == 2.0
    assert webull_auth_status()["rate_limited"] is True


def test_migrate_legacy_webull_token_from_conf(tmp_path, monkeypatch):
    legacy_dir = tmp_path / "conf"
    legacy_dir.mkdir()
    legacy_token = legacy_dir / "token.txt"
    legacy_token.write_text("abc123\n1700000000\nNORMAL\n", encoding="utf-8")
    target_dir = tmp_path / "webull"

    monkeypatch.chdir(tmp_path)
    assert migrate_legacy_webull_token(target_dir) is True
    assert (target_dir / "token.txt").read_text(encoding="utf-8") == legacy_token.read_text(encoding="utf-8")


def test_configure_data_paths_sets_webull_token_dir(tmp_path, monkeypatch):
    import gex_core.data_root as data_root

    data_root._CONFIGURED = False
    monkeypatch.setenv("GEX_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("WEBULL_OPENAPI_TOKEN_DIR", raising=False)

    root = data_root.configure_data_paths()
    token_dir = Path(os.environ["WEBULL_OPENAPI_TOKEN_DIR"])
    assert token_dir == root / "webull"
    assert token_dir.is_dir()


def test_webull_trade_endpoint_migrates_deprecated_host(monkeypatch):
    monkeypatch.setenv("GEX_WEBULL_ENDPOINT", "us-openapi.webullbroker.com")
    monkeypatch.setenv("GEX_WEBULL_DATA_ENDPOINT", "us-openapi.webullbroker.com")
    monkeypatch.delenv("GEX_WEBULL_USE_UAT", raising=False)
    assert webull_trade_endpoint() == "api.webull.com"
    assert webull_data_endpoint() == "broker-api.webull.com"
