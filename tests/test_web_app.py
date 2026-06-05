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
    # Soft, reassuring message — not the alarming "Check service logs" banner.
    assert b"Showing the last saved snapshot" in response.data
    assert b"Snapshot refresh failed for SPX and no saved snapshot" not in response.data
