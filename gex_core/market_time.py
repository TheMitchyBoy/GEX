"""US market timezone helpers for Periscope timestamps and session dates."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from gex_core.exports import parse_timestamp

MARKET_TZ_NAME = os.environ.get("GEX_MARKET_TIMEZONE", "America/New_York")
MARKET_TZ = ZoneInfo(MARKET_TZ_NAME)
MARKET_TZ_LABEL = os.environ.get("GEX_MARKET_TIMEZONE_LABEL", "ET")


def parse_export_ts_utc(ts: str) -> datetime:
    """Parse an export key ``YYYY-MM-DD_HHMMSS`` as UTC."""
    return parse_timestamp(ts).replace(tzinfo=timezone.utc)


def ts_market_date(ts: str) -> str:
    """Trading session calendar date for an export timestamp key."""
    return parse_export_ts_utc(ts).astimezone(MARKET_TZ).date().isoformat()


def ts_market_time_label(ts: str, *, with_tz: bool = True) -> str:
    """Clock time in the market timezone (e.g. ``09:30 ET``)."""
    local = parse_export_ts_utc(ts).astimezone(MARKET_TZ)
    label = local.strftime("%H:%M")
    if with_tz:
        return f"{label} {MARKET_TZ_LABEL}"
    return label


def ts_display_label(ts: str) -> str:
    """Full display label for dashboards (market-local)."""
    local = parse_export_ts_utc(ts).astimezone(MARKET_TZ)
    return f"{local.strftime('%Y-%m-%d %H:%M:%S')} {MARKET_TZ_LABEL}"


def market_today() -> str:
    """Current trading calendar date in the market timezone."""
    return datetime.now(MARKET_TZ).date().isoformat()


def market_now_export_ts() -> str:
    """Export-style key for the current instant (UTC, storage-compatible)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def _parse_iso_ts(ts: str) -> datetime:
    value = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def bars_held_since_entry(entry_ts: str, *, bar_minutes: float) -> int:
    """Gamma bars elapsed since trade entry (for time-stop semantics)."""
    if bar_minutes <= 0:
        return 0
    try:
        entry = _parse_iso_ts(entry_ts)
    except (TypeError, ValueError):
        return 0
    elapsed = datetime.now(timezone.utc) - entry
    return max(0, int(elapsed.total_seconds() // (bar_minutes * 60)))


def minutes_between_timestamps(start_ts: str, end_ts: str) -> float | None:
    """Wall-clock minutes between export snapshot timestamps."""
    try:
        from gex_core.exports import parse_timestamp

        start = parse_timestamp(start_ts)
        end = parse_timestamp(end_ts)
    except (TypeError, ValueError):
        return None
    return max(0.0, (end - start).total_seconds() / 60.0)


def bars_between_timestamps(entry_ts: str, current_ts: str, *, bar_minutes: float) -> int:
    """Gamma bars elapsed between two snapshot timestamps."""
    minutes = minutes_between_timestamps(entry_ts, current_ts)
    if minutes is None or bar_minutes <= 0:
        return 0
    return max(0, int(minutes // bar_minutes))


def parse_export_ts_market(ts: str) -> datetime:
    """Export key instant in the market timezone."""
    return parse_export_ts_utc(ts).astimezone(MARKET_TZ)


def is_trading_weekday(*, now: datetime | None = None) -> bool:
    """True Mon–Fri in the market timezone."""
    anchor = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    return anchor.weekday() < 5


def _observed_holiday(day: date) -> date:
    """NYSE-style weekend observation (Sat -> Fri, Sun -> Mon)."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, n: int, weekday: int) -> date:
    """Nth weekday in a month (n=1 first, n=-1 last)."""
    if n == 0:
        raise ValueError("n must be non-zero")
    if n > 0:
        day = date(year, month, 1)
        while day.weekday() != weekday:
            day += timedelta(days=1)
        return day + timedelta(weeks=n - 1)
    day = date(year, month + 1, 1) - timedelta(days=1)
    while day.weekday() != weekday:
        day -= timedelta(days=1)
    return day + timedelta(weeks=n + 1)


def _easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (Anonymous algorithm)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def us_equity_holidays(year: int) -> set[date]:
    """US equity market closed dates for a calendar year (NYSE-style)."""
    easter = _easter_sunday(year)
    holidays = {
        _observed_holiday(date(year, 1, 1)),
        _nth_weekday(year, 1, 1, 0),
        _nth_weekday(year, 2, 3, 0),
        easter - timedelta(days=2),
        _nth_weekday(year, 5, -1, 0),
        _observed_holiday(date(year, 6, 19)),
        _observed_holiday(date(year, 7, 4)),
        _nth_weekday(year, 9, 1, 0),
        _nth_weekday(year, 11, 4, 3),
        _observed_holiday(date(year, 12, 25)),
    }
    return holidays


def _parse_market_calendar_date(value: date | str | datetime) -> date:
    if isinstance(value, datetime):
        return value.astimezone(MARKET_TZ).date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def is_equity_trading_day(value: date | str | datetime) -> bool:
    """True on weekdays that are not US equity market holidays."""
    day = _parse_market_calendar_date(value)
    if day.weekday() >= 5:
        return False
    return day not in us_equity_holidays(day.year)


def export_ts_is_trading_day(ts: str) -> bool:
    """True when the export timestamp falls on an equity trading session date."""
    try:
        return is_equity_trading_day(parse_export_ts_market(ts).date())
    except (TypeError, ValueError):
        return False


def export_ts_is_trading_session(ts: str) -> bool:
    """True during weekday regular session hours for an export timestamp."""
    return is_trader_session_active(now=parse_export_ts_market(ts))


def filter_trading_history(
    history: list[dict],
    *,
    session_only: bool = True,
    intraday_only: bool = False,
) -> list[dict]:
    """Drop off-session snapshots from walk-forward history."""
    rows = list(history)
    if session_only:
        rows = [row for row in rows if export_ts_is_trading_day(str(row.get("ts") or ""))]
    if intraday_only:
        rows = [row for row in rows if export_ts_is_trading_session(str(row.get("ts") or ""))]
    return rows


def is_trader_session_active(*, now: datetime | None = None) -> bool:
    """True during configured US equity session (weekdays, market hours ET)."""
    if os.environ.get("GEX_TRADER_SESSION_ONLY", "1").strip().lower() in {"0", "false", "no", "off"}:
        return True
    anchor = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    if anchor.weekday() >= 5:
        return False
    try:
        open_h = int(os.environ.get("GEX_TRADER_SESSION_OPEN_HOUR", "9"))
        open_m = int(os.environ.get("GEX_TRADER_SESSION_OPEN_MIN", "30"))
        close_h = int(os.environ.get("GEX_TRADER_SESSION_CLOSE_HOUR", "16"))
        close_m = int(os.environ.get("GEX_TRADER_SESSION_CLOSE_MIN", "0"))
    except (TypeError, ValueError):
        open_h, open_m, close_h, close_m = 9, 30, 16, 0
    open_minutes = open_h * 60 + open_m
    close_minutes = close_h * 60 + close_m
    now_minutes = anchor.hour * 60 + anchor.minute
    return open_minutes <= now_minutes < close_minutes


def _session_minutes(*, now: datetime | None = None) -> tuple[int, int, int]:
    anchor = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    try:
        open_h = int(os.environ.get("GEX_TRADER_SESSION_OPEN_HOUR", "9"))
        open_m = int(os.environ.get("GEX_TRADER_SESSION_OPEN_MIN", "30"))
        close_h = int(os.environ.get("GEX_TRADER_SESSION_CLOSE_HOUR", "16"))
        close_m = int(os.environ.get("GEX_TRADER_SESSION_CLOSE_MIN", "0"))
    except (TypeError, ValueError):
        open_h, open_m, close_h, close_m = 9, 30, 16, 0
    open_minutes = open_h * 60 + open_m
    close_minutes = close_h * 60 + close_m
    now_minutes = anchor.hour * 60 + anchor.minute
    return now_minutes, open_minutes, close_minutes


def is_entry_window_active(*, now: datetime | None = None) -> bool:
    """True during entry-friendly session window (skip open chop / close decay)."""
    from gex_core.trading.config import (
        entry_time_filter_enabled,
        entry_window_after_open_min,
        entry_window_before_close_min,
    )

    if not entry_time_filter_enabled():
        return is_trader_session_active(now=now)
    anchor = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    if anchor.weekday() >= 5:
        return False
    now_minutes, open_minutes, close_minutes = _session_minutes(now=anchor)
    earliest = open_minutes + entry_window_after_open_min()
    latest = close_minutes - entry_window_before_close_min()
    return earliest <= now_minutes < latest


def is_eod_flatten_time(*, now: datetime | None = None) -> bool:
    from gex_core.trading.config import eod_flatten_enabled, eod_flatten_hour, eod_flatten_minute

    if not eod_flatten_enabled():
        return False
    anchor = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    if anchor.weekday() >= 5:
        return False
    flatten_minutes = eod_flatten_hour() * 60 + eod_flatten_minute()
    now_minutes = anchor.hour * 60 + anchor.minute
    return now_minutes >= flatten_minutes


def export_ts_entry_window_ok(ts: str) -> bool:
    """Entry window check for walk-forward backtests using export timestamps."""
    from gex_core.trading.config import entry_time_filter_enabled

    if not export_ts_is_trading_day(ts):
        return False
    local = parse_export_ts_market(ts)
    if not entry_time_filter_enabled():
        return is_trader_session_active(now=local)
    return is_entry_window_active(now=local)


def export_ts_eod_flatten(ts: str) -> bool:
    local = parse_export_ts_market(ts)
    return is_eod_flatten_time(now=local)
