def test_ticker_page_returns_200_with_history():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/ticker/SPX")
    assert response.status_code == 200
    assert b"Market Maker Exposure" in response.data
    assert b"GEX Assistant" in response.data


def test_index_renders_periscope_dashboard():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"PERISCOPE" in response.data
    assert b"10 min" in response.data
    assert b"sessionDate" in response.data


def test_api_periscope_returns_json():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/api/periscope?exposure=gamma")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ticker"] == "SPX"
    assert payload["exposure"] == "gamma"


def test_api_agent_analyze_returns_json():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/api/agent/analyze?exposure=gamma")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ticker"] == "SPX"
    assert "who" in payload
    assert "narrative" in payload


def test_api_chat_requires_message():
    import web_app

    client = web_app.APP.test_client()
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 400


def test_api_chat_rule_based_reply(monkeypatch):
    import web_app

    monkeypatch.setattr(
        web_app,
        "chat_reply",
        lambda **kwargs: {
            "session_id": "test-session",
            "reply": "LONG gamma environment.",
            "llm_source": "rule_based",
            "uw_data_fed": False,
            "messages": [],
        },
    )
    client = web_app.APP.test_client()
    response = client.post("/api/chat", json={"message": "What regime are we in?"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["reply"] == "LONG gamma environment."


def test_api_agent_predict_returns_503_without_live_uw(monkeypatch):
    import web_app

    monkeypatch.setattr(web_app, "get_uw_data", lambda *a, **k: None)
    client = web_app.APP.test_client()
    response = client.get("/api/agent/predict")
    assert response.status_code == 503
    payload = response.get_json()
    assert "error" in payload


def test_force_refresh_failure_with_history_degrades_to_stale(monkeypatch):
    """A failed forced refresh must not show the hard error when cached data exists."""
    import web_app

    monkeypatch.setattr(web_app, "refresh_ticker", lambda *a, **k: False)
    monkeypatch.setattr(web_app, "refresh_uw_data", lambda *a, **k: None)

    client = web_app.APP.test_client()
    response = client.get("/ticker/SPX/?force_refresh=1")
    assert response.status_code == 200
    # Soft, reassuring message — not the alarming hard-failure banner.
    assert b"Showing last saved snapshot" in response.data
    assert b"check service logs" not in response.data


def test_classify_uw_error_categories():
    import requests
    import web_app

    assert web_app._classify_uw_error(EnvironmentError("no key")) == "not_configured"

    http_403 = requests.HTTPError()
    http_403.response = type("R", (), {"status_code": 403})()
    assert web_app._classify_uw_error(http_403) == "auth"

    http_429 = requests.HTTPError()
    http_429.response = type("R", (), {"status_code": 429})()
    assert web_app._classify_uw_error(http_429) == "rate_limited"

    assert web_app._classify_uw_error(requests.Timeout()) == "network"
    assert web_app._classify_uw_error(ValueError("weird")) == "error"


def test_uw_failure_reason_not_configured(monkeypatch):
    import web_app

    monkeypatch.setenv("UW_API_KEY", "")
    assert web_app._uw_failure_reason("SPX") == "not_configured"


def test_force_refresh_without_uw_key_skips_fetch(monkeypatch):
    import web_app

    monkeypatch.setenv("UW_API_KEY", "")
    monkeypatch.setattr(web_app, "uw_api_configured", lambda: False)
    calls = {"csv": 0, "uw": 0}

    def _csv_refresh(*_a, **_k):
        calls["csv"] += 1
        return False

    def _uw_refresh(*_a, **_k):
        calls["uw"] += 1
        return None

    monkeypatch.setattr(web_app, "refresh_ticker", _csv_refresh)
    monkeypatch.setattr(web_app, "refresh_uw_data", _uw_refresh)

    client = web_app.APP.test_client()
    response = client.get("/ticker/SPX/?force_refresh=1")

    assert response.status_code == 200
    assert calls == {"csv": 0, "uw": 0}
    assert b"last saved snapshot" in response.data or b"Live data isn't configured" in response.data


def test_persistent_banner_when_uw_not_configured(monkeypatch):
    import web_app

    monkeypatch.setenv("UW_API_KEY", "")
    monkeypatch.setattr(web_app, "uw_api_configured", lambda: False)
    client = web_app.APP.test_client()
    response = client.get("/ticker/SPX/")
    assert response.status_code == 200
    assert b"UW_API_KEY" in response.data
