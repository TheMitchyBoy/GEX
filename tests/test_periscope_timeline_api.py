def test_api_periscope_timeline_returns_dates_and_slices():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/api/periscope/timeline")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ticker"] == "SPX"
    assert "dates" in payload
    assert "slices_by_date" in payload
    assert "timeline" in payload
