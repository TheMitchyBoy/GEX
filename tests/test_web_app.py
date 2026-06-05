def test_ticker_page_returns_200_with_history():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/ticker/SPX")
    assert response.status_code == 200
