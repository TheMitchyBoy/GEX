from datetime import date

from gex_core.market_time import is_equity_trading_day, us_equity_holidays


def test_good_friday_2026_is_not_trading_day():
    assert date(2026, 4, 3) in us_equity_holidays(2026)
    assert not is_equity_trading_day("2026-04-03")


def test_regular_weekday_is_trading_day():
    assert is_equity_trading_day("2026-06-05")


def test_weekend_is_not_trading_day():
    assert not is_equity_trading_day("2026-06-06")
