"""Wall GEX dashboard API and engine status."""

from unittest.mock import patch

import pandas as pd

from gex_core.trading.low_gex_engine import run_low_gex_trade, wall_gex_status


def test_wall_gex_status_defaults():
    status = wall_gex_status("SPX")
    assert status["ticker"] == "SPX"
    assert status["stop_loss_pct"] == 0.03
    assert status["take_profit_pct"] == 0.22
    assert status["max_hold_bars"] == 8
    assert "open_positions" in status
    assert "performance" in status
    assert "webull_auth" in status


def test_run_low_gex_trade_requires_arm_when_execute():
    exposure = pd.Series([-2.0, 0.5, 1.0], index=[7440.0, 7460.0, 7480.0])
    with (
        patch("gex_core.trading.low_gex_engine.is_trader_session_active", return_value=True),
        patch("gex_core.trading.low_gex_engine.is_entry_window_active", return_value=True),
        patch("gex_core.trading.low_gex_engine.is_trader_armed", return_value=False),
        patch("gex_core.trading.low_gex_engine.manage_wall_gex_exits", return_value={"eod_exits": [], "exits": []}),
    ):
        result = run_low_gex_trade(
            ticker="SPX",
            spot=7460.0,
            exposure=exposure,
            execute=True,
            force=False,
        )
    assert result["ran"] is False
    assert "disarmed" in result.get("reason", "").lower()


def test_api_wall_gex_status():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/api/wall-gex/status?ticker=SPX")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ticker"] == "SPX"
    assert "status" in payload
    assert payload["status"]["take_profit_pct"] == 0.22


def test_api_wall_gex_status_returns_json_on_error(monkeypatch):
    from web_app import APP

    def _boom(*_a, **_k):
        raise RuntimeError("context failed")

    monkeypatch.setattr("web_app._wall_gex_live_data", _boom)
    client = APP.test_client()
    response = client.get("/api/wall-gex/status?ticker=SPX")
    assert response.status_code == 200
    assert response.content_type.startswith("application/json")
    payload = response.get_json()
    assert payload["status"]["ticker"] == "SPX"
    assert "context failed" in payload["last_cycle"]["reason"]


def test_api_unknown_returns_json_not_html():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/api/wall-gex/missing")
    assert response.status_code == 404
    assert response.content_type.startswith("application/json")
    assert response.get_json()["error"]


def test_api_wall_gex_reset_token_deletes_file_and_disarms(tmp_path, monkeypatch):
    from web_app import APP

    monkeypatch.setenv("GEX_TRADER_PAPER", "0")
    monkeypatch.setenv("GEX_WEBULL_APP_KEY", "key")
    monkeypatch.setenv("GEX_WEBULL_APP_SECRET", "secret")
    monkeypatch.setenv("GEX_WEBULL_ACCOUNT_ID", "acct-1")
    monkeypatch.setenv("WEBULL_OPENAPI_TOKEN_DIR", str(tmp_path))
    token_file = tmp_path / "token.txt"
    token_file.write_text("old\n1\nEXPIRED\n", encoding="utf-8")

    with patch("gex_core.trading.webull_broker.fetch_account_balance", return_value={"code": 0, "data": {}}):
        client = APP.test_client()
        arm = client.post("/api/wall-gex/arm", json={"armed": True, "ticker": "SPX", "live_confirm": True})
        assert arm.get_json()["armed"] is True
        response = client.post("/api/wall-gex/reset-token", json={"ticker": "SPX", "trigger_sms": True})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["removed"] is True
    assert payload["armed"] is False
    assert not token_file.exists()
    assert payload["probe_triggered"] is True


def test_api_wall_gex_arm_disarm():
    from web_app import APP

    client = APP.test_client()
    arm = client.post("/api/wall-gex/arm", json={"armed": True, "ticker": "SPX"})
    assert arm.status_code == 200
    assert arm.get_json()["armed"] is True
    disarm = client.post("/api/wall-gex/arm", json={"armed": False, "ticker": "SPX"})
    assert disarm.status_code == 200
    assert disarm.get_json()["armed"] is False
