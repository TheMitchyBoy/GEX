from gex_core.uw_price_stream import (
    UWPriceStream,
    _is_expected_disconnect,
    _parse_price_message,
    _should_reset_reconnect_attempt,
)


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


def test_is_expected_disconnect_recognizes_remote_drop():
    class FakeClosed(Exception):
        pass

    exc = FakeClosed("no close frame received or sent")
    assert _is_expected_disconnect(exc) is True
    assert _is_expected_disconnect(ConnectionResetError()) is True
    assert _is_expected_disconnect(RuntimeError("auth failed")) is False


def test_should_reset_reconnect_attempt_only_after_stable_session():
    assert _should_reset_reconnect_attempt(59.9) is False
    assert _should_reset_reconnect_attempt(60.0) is True
    assert _should_reset_reconnect_attempt(3600.0) is True
