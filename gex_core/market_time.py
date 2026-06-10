"""US market timezone helpers for Periscope timestamps and session dates."""

from __future__ import annotations

import os
from datetime import datetime, timezone
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


def export_ts_is_trading_day(ts: str) -> bool:
    """True when the export timestamp falls on a weekday (market calendar)."""
    try:
        return is_trading_weekday(now=parse_export_ts_market(ts))
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
