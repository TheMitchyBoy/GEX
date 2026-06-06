from gex_core.market_time import ts_display_label, ts_market_date, ts_market_time_label


def test_ts_market_date_converts_utc_key_to_et_session():
    # 01:22 UTC on June 6 is still June 5 evening in US/Eastern (EDT).
    assert ts_market_date("2026-06-06_012248") == "2026-06-05"


def test_ts_market_time_label_shows_et_clock():
    label = ts_market_time_label("2026-06-06_012248")
    assert label.endswith("ET")
    assert "21:22" in label or "20:22" in label


def test_ts_display_label_includes_timezone():
    label = ts_display_label("2026-06-05_143000")
    assert "2026-06-05" in label
    assert "ET" in label
