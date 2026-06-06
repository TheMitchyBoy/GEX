from gex_core.uw_price_stream import UWPriceStream, _parse_price_message


def test_parse_price_message_extracts_close_and_time():
    point = _parse_price_message(
        "price:SPX",
        {"close": "5012.5", "time": 1726670327692, "vol": 123},
    )
    assert point is not None
    assert point["ticker"] == "SPX"
    assert point["close"] == 5012.5
    assert "2024" in point["ts"]


def test_stream_ingest_and_read_latest():
    stream = UWPriceStream()
    stream.ingest_point(
        "SPX",
        {"ticker": "SPX", "ts": "2026-06-05T15:00:00+00:00", "close": 5000.0},
    )
    assert stream.get_latest_price("SPX") == 5000.0
    points = stream.get_price_points("SPX")
    assert points == [{"ts": "2026-06-05T15:00:00+00:00", "close": 5000.0}]


def test_parse_price_message_ignores_join_ack():
    assert _parse_price_message("price:SPX", {"status": "ok", "response": {}}) is None
