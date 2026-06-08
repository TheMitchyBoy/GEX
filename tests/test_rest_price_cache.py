import time

from gex_core import market_features as mf


def test_cached_rest_price_throttles_uw_calls(monkeypatch):
    mf._REST_PRICE_CACHE.clear()
    calls = {"n": 0}

    def _fake_stock_state(ticker, api_key=None):
        calls["n"] += 1
        return 5000.0

    monkeypatch.setattr(mf, "_uw_price_enabled", lambda: True)
    monkeypatch.setattr("gex_core.uw_price_stream.get_uw_price_stream", lambda: type("S", (), {"get_latest_price": lambda self, t: 0.0})())
    monkeypatch.setattr("gex_core.uw_loader.fetch_uw_stock_state_price", _fake_stock_state)
    monkeypatch.setattr("gex_core.uw_loader.fetch_uw_spot", lambda *a, **k: 0.0)
    monkeypatch.setenv("GEX_UW_REST_PRICE_CACHE_SECONDS", "60")

    assert mf.fetch_spx_price("SPX") == 5000.0
    assert mf.fetch_spx_price("SPX") == 5000.0
    assert calls["n"] == 1

    mf._REST_PRICE_CACHE.clear()
