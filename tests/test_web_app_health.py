def test_health_endpoint_always_live():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ticker"] == "SPX"
    assert "history_depth" in payload
    assert "ready" in payload
    assert "healthy" in payload


def test_health_ready_strict():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/health/ready")
    assert response.status_code in {200, 503}
    payload = response.get_json()
    assert payload["ticker"] == "SPX"


def test_api_latest_summary():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/api/latest-summary")
    assert response.status_code == 200
    data = response.get_json()
    assert data["ticker"] == "SPX"
