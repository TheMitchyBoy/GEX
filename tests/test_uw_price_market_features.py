from gex_core.market_features import fetch_spx_price_series_for_dashboard
from gex_core.uw_price_stream import UWPriceStream


def test_dashboard_price_series_prefers_uw_websocket(monkeypatch):
    stream = UWPriceStream()
    stream.ingest_point(
        "SPX",
        {"ticker": "SPX", "ts": "2026-06-05T20:00:00+00:00", "close": 5015.0},
    )
    monkeypatch.setattr("gex_core.market_features._uw_price_enabled", lambda: True)
    monkeypatch.setattr("gex_core.uw_price_stream.get_uw_price_stream", lambda: stream)
    monkeypatch.setattr(
        "gex_core.market_features.fetch_uw_price_history",
        lambda ticker, **kwargs: [
            {"ts": "2026-06-05T15:00:00+00:00", "close": 5000.0},
            {"ts": "2026-06-05T19:00:00+00:00", "close": 5010.0},
        ],
    )
    monkeypatch.setattr(
        "gex_core.storage.fetch_index_spot_series",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr("gex_core.market_features.fetch_spx_price", lambda ticker="SPX": 5015.0)

    points, current, source = fetch_spx_price_series_for_dashboard("SPX")
    assert current == 5015.0
    assert source == "uw-live"
    assert points[-1]["close"] == 5015.0
    assert points[0]["close"] == 5000.0
