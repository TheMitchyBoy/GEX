def test_ticker_page_returns_200_with_history():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/ticker/SPX")
    assert response.status_code == 200
    assert b"0DTE Movement Priority" in response.data


def test_force_refresh_failure_with_history_degrades_to_stale(monkeypatch):
    """A failed forced refresh must not show the hard error when cached data exists."""
    import web_app

    monkeypatch.setattr(web_app, "refresh_ticker", lambda *a, **k: False)
    monkeypatch.setattr(web_app, "refresh_uw_data", lambda *a, **k: None)

    client = web_app.APP.test_client()
    response = client.get("/ticker/SPX/?force_refresh=1")
    assert response.status_code == 200
    # Soft, reassuring message — not the alarming hard-failure banner.
    assert b"Showing the last saved snapshot" in response.data
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

    monkeypatch.setattr(web_app, "_UW_ENABLED", False)
    assert web_app._uw_failure_reason("SPX") == "not_configured"
