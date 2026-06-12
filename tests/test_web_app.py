def test_ticker_page_returns_200_with_history():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/ticker/SPX")
    assert response.status_code == 200
    assert b"Wall GEX Trader" in response.data


def test_index_renders_wall_gex_dashboard():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Wall GEX Trader" in response.data
    assert b"/api/wall-gex/status" in response.data


def test_ticker_page_does_not_block_on_live_wall_gex_data(monkeypatch):
    from web_app import APP

    calls = {"n": 0}

    def _track(*_a, **_k):
        calls["n"] += 1
        return 6000.0, None, {"ran": False}

    monkeypatch.setattr("web_app._wall_gex_live_data", _track)
    client = APP.test_client()
    response = client.get("/ticker/SPX/")
    assert response.status_code == 200
    assert b"Loading signal" in response.data
    assert calls["n"] == 0


def test_api_agent_daily_strategy():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/api/agent/daily-strategy")
    assert response.status_code == 200
    data = response.get_json()
    assert data["ticker"] == "SPX"
    assert "strategy" in data
    assert "recent_lessons" in data


def test_ticker_api_payload_skips_backtest_when_configured(monkeypatch):
    from web_app import _ticker_api_payload

    calls = {"n": 0}

    def _track(*_a, **_k):
        calls["n"] += 1
        return {"n": 0, "accuracy": None}

    monkeypatch.setenv("GEX_DASHBOARD_SKIP_BACKTEST", "1")
    monkeypatch.setattr("web_app.backtest_delta_sign_accuracy", _track)
    payload = _ticker_api_payload("SPX")
    assert payload["ticker"] == "SPX"
    assert calls["n"] == 0


def test_gamma_dashboard_renders_periscope():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/gamma")
    assert response.status_code == 200
    assert b"Gamma Magnet Strategy" in response.data
    assert b"strategyChart" in response.data


def test_gamma_near_dashboard_renders_near_spot_walls():
    from web_app import APP

    client = APP.test_client()
    response = client.get("/gamma/near")
    assert response.status_code == 200
    assert b"Near-Spot Walls" in response.data
    assert b"Wall GEX" in response.data
    assert b"1.0% strikes" in response.data
    assert b"Full gamma view" in response.data
    assert b"strikeWindowPct" in response.data
    assert b"wallMode" in response.data


def test_api_trader_strategy_honors_window_pct(monkeypatch):
    from web_app import APP

    captured: dict = {}

    def _fake_wall_build(**kwargs):
        captured.update(kwargs)
        return {"state": {"signals": {}}, "chart_json": "{}", "window_pct": kwargs.get("window_pct"), "strategy_mode": "wall"}

    monkeypatch.setattr("gex_core.trading.strategy_viz.build_wall_strategy_dashboard", _fake_wall_build)
    monkeypatch.setattr("web_app.build_periscope_context", lambda **_k: {"spot": 6000.0, "selected": {}, "history": []})
    monkeypatch.setattr("web_app._strategy_exposure_from_context", lambda _ctx: (None, None))
    monkeypatch.setattr("web_app._uw_bundle_for_context", lambda **_k: None)
    monkeypatch.setattr("web_app.get_uw_data_with_timeout", lambda _t, **_k: None)
    monkeypatch.setattr("web_app._uw_live_enabled", lambda: False)

    client = APP.test_client()
    response = client.get("/api/trader/strategy?window_pct=0.01")
    assert response.status_code == 200
    assert captured["window_pct"] == 0.01
    assert captured["max_strikes"] == 40
    assert response.get_json().get("strategy_mode") == "wall"


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

    monkeypatch.setattr(web_app, "get_uw_data_with_timeout", lambda *a, **k: None)
    client = web_app.APP.test_client()
    response = client.get("/api/agent/predict")
    assert response.status_code == 503
    payload = response.get_json()
    assert "error" in payload


def test_get_force_refresh_query_is_ignored(monkeypatch):
    """GET force_refresh must not trigger UW fetches (crawler-safe)."""
    import web_app

    calls = {"refresh": 0}

    def _fake_refresh(*_a, **_k):
        calls["refresh"] += 1
        return False, "error"

    monkeypatch.setattr(web_app, "_run_ticker_refresh", _fake_refresh)

    client = web_app.APP.test_client()
    response = client.get("/ticker/SPX/?force_refresh=1")
    assert response.status_code == 200
    assert calls["refresh"] == 0


def test_post_refresh_failure_with_history_degrades_to_stale(monkeypatch):
    """A failed authorized refresh must not show the hard error when cached data exists."""
    import web_app

    monkeypatch.setenv("GEX_ADMIN_TOKEN", "test-admin-token")
    monkeypatch.setattr(web_app, "refresh_ticker", lambda *a, **k: False)
    monkeypatch.setattr(web_app, "refresh_uw_data", lambda *a, **k: None)

    client = web_app.APP.test_client()
    response = client.post(
        "/ticker/SPX/refresh",
        headers={"X-Admin-Token": "test-admin-token"},
    )
    assert response.status_code == 302
    follow = client.get(response.headers["Location"])
    assert follow.status_code == 200
    assert b"Wall GEX Trader" in follow.data


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


def test_post_refresh_without_token_is_forbidden(monkeypatch):
    import web_app

    monkeypatch.setenv("GEX_ADMIN_TOKEN", "test-admin-token")
    client = web_app.APP.test_client()
    response = client.post("/ticker/SPX/refresh")
    assert response.status_code == 403


def test_post_refresh_without_uw_key_skips_fetch(monkeypatch):
    import web_app

    monkeypatch.setenv("GEX_ADMIN_TOKEN", "test-admin-token")
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
    response = client.post(
        "/ticker/SPX/refresh",
        headers={"X-Admin-Token": "test-admin-token"},
    )

    assert response.status_code == 302
    assert calls == {"csv": 0, "uw": 0}
    follow = client.get(response.headers["Location"])
    assert follow.status_code == 200
    assert b"Wall GEX Trader" in follow.data


def test_live_uw_failure_shows_stale_banner_on_latest_slice(monkeypatch):
    import web_app

    monkeypatch.setenv("UW_API_KEY", "dummy-key")
    monkeypatch.setattr(web_app, "uw_api_configured", lambda: True)
    monkeypatch.setattr(web_app, "_uw_live_enabled", lambda: True)
    monkeypatch.setattr(web_app, "get_uw_data_with_timeout", lambda *_a, **_k: None)
    monkeypatch.setattr(web_app, "_uw_failure_reason", lambda *_a, **_k: "rate_limited")

    client = web_app.APP.test_client()
    response = client.get("/ticker/SPX/gamma")
    assert response.status_code == 200
    assert b"Showing last saved snapshot" in response.data
    assert b"rate-limiting" in response.data


def test_persistent_banner_when_uw_not_configured(monkeypatch):
    import web_app

    monkeypatch.setenv("UW_API_KEY", "")
    monkeypatch.setattr(web_app, "uw_api_configured", lambda: False)
    client = web_app.APP.test_client()
    response = client.get("/ticker/SPX/gamma")
    assert response.status_code == 200
    assert b"UW_API_KEY" in response.data


def test_spot_stream_does_not_poll_uw_rest(monkeypatch):
    """SSE spot updates must use websocket cache only (no REST every 0.5s)."""
    from unittest.mock import MagicMock, patch

    import web_app
    from gex_core.uw_price_stream import UWPriceStream

    stream = UWPriceStream()
    stream.ingest_point(
        "SPX",
        {"ticker": "SPX", "ts": "2026-06-05T20:00:00+00:00", "close": 5012.5},
    )
    monkeypatch.setattr("gex_core.uw_price_stream.get_uw_price_stream", lambda: stream)
    monkeypatch.setenv("GEX_SPOT_STREAM_POLL_SECONDS", "0.25")

    client = web_app.APP.test_client()
    with patch("gex_core.market_features.fetch_spx_price") as mock_rest:
        response = client.get("/api/spot-stream?ticker=SPX")
        assert response.status_code == 200
        # Read one event from the stream without blocking forever.
        chunks = []
        for _ in range(3):
            chunks.append(next(response.response))
            if b"5012.5" in chunks[-1]:
                break
        mock_rest.assert_not_called()
        assert any(b"5012.5" in chunk for chunk in chunks)


def test_refresh_uw_data_does_not_compute_gamma_flip(monkeypatch):
    """Live UW cache skips gamma flip (not shown on dashboard)."""
    from unittest.mock import patch

    import pandas as pd

    import web_app
    from gex_core.pipeline import GexAggregates
    from gex_core.spot_exposure import spot_exposure_net_series

    monkeypatch.setenv("UW_API_KEY", "test-key")
    monkeypatch.setattr(web_app, "uw_api_configured", lambda: True)
    web_app._UW_CACHE.clear()

    greek_df = pd.DataFrame(
        {
            "strike": [7440.0, 7450.0, 7460.0, 7480.0],
            "call_gex": [2.0, 2.5, 3.0, 2.0],
            "put_gex": [-3.0, -0.5, -1.0, -1.0],
            "net_gex": [-1.0, 2.0, 2.0, 1.0],
        }
    )
    spot_df = pd.DataFrame(
        {
            "strike": [7440.0, 7450.0, 7460.0, 7480.0],
            "call_gamma_oi": [1e9, 2e9, 3e9, 2e9],
            "put_gamma_oi": [-3e9, -1.5e9, -1e9, -1e9],
        }
    )
    spot_df["net_gamma_oi"] = spot_df["call_gamma_oi"] + spot_df["put_gamma_oi"]
    gex_by_strike = spot_exposure_net_series(spot_df, "gamma")
    gex_by_strike.attrs["greek_exposure_df"] = greek_df
    gex_by_strike.attrs["spot_exposures_df"] = spot_df
    gex_by_strike.attrs["uw_endpoint"] = "spot-exposures/strike"
    agg = GexAggregates(
        gex_by_strike=gex_by_strike,
        gex_by_expiration=pd.Series(dtype=float),
        cumulative_gex=gex_by_strike.cumsum(),
        surface_data=pd.DataFrame(),
        total_gex_bn=float(gex_by_strike.sum()),
    )

    with (
        patch("gex_core.uw_loader.fetch_uw_gex", return_value=(7460.0, agg)),
        patch("gex_core.uw_loader.fetch_spot_gamma_aggregate_bn", return_value=1.0),
        patch("gex_core.ai_analyst.analyze_dealer_gamma", return_value=None),
    ):
        entry = web_app.refresh_uw_data("SPX", force=True)

    assert entry is not None
    assert entry["gamma_flip"] is None
