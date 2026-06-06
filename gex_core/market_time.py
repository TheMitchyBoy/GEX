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
