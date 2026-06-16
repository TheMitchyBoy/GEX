"""Market-hours and adaptive refresh scheduling for the processor."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from gex_core.env_bootstrap import parse_env_minutes
from gex_core.market_time import MARKET_TZ, is_trader_session_active, is_trading_weekday
from gex_core.refresh import DEFAULT_REFRESH_MINUTES


def processor_refresh_enabled(*, now: datetime | None = None) -> bool:
    """Return False when the processor should skip a refresh cycle."""
    if os.environ.get("GEX_PROCESSOR_REFRESH_ALWAYS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    anchor = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    if not is_trading_weekday(now=anchor):
        return os.environ.get("GEX_REFRESH_WEEKENDS", "").strip().lower() in {"1", "true", "yes", "on"}
    if not is_trader_session_active(now=anchor):
        return os.environ.get("GEX_REFRESH_OFFHOURS", "").strip().lower() in {"1", "true", "yes", "on"}
    return True


def adaptive_refresh_minutes(*, now: datetime | None = None) -> float:
    """Shorter interval during active session / elevated vol; longer off-hours."""
    base = parse_env_minutes("GEX_REFRESH_INTERVAL_MINUTES", DEFAULT_REFRESH_MINUTES)
    anchor = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    if not is_trading_weekday(now=anchor):
        return parse_env_minutes("GEX_OFFHOURS_REFRESH_MINUTES", max(base, 30.0))
    if not is_trader_session_active(now=anchor):
        return parse_env_minutes("GEX_OFFHOURS_REFRESH_MINUTES", max(base, 20.0))

    try:
        from gex_core.market_context_cache import cached_vol_regime

        vix = float((cached_vol_regime() or {}).get("vix_level") or 0.0)
    except Exception:
        vix = 0.0
    event_risk = 0.0
    if vix >= 28:
        return max(2.0, base / 3.0)
    if vix >= 22:
        return max(3.0, base / 2.0)
    return base


def should_refresh_now(
    last_refresh_at: datetime | None,
    *,
    now: datetime | None = None,
) -> bool:
    """True when enough time has elapsed for the current adaptive interval."""
    if not processor_refresh_enabled(now=now):
        return False
    if last_refresh_at is None:
        return True
    interval = adaptive_refresh_minutes(now=now)
    anchor = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    last = last_refresh_at.astimezone(timezone.utc) if last_refresh_at.tzinfo else last_refresh_at.replace(tzinfo=timezone.utc)
    return (anchor - last) >= timedelta(minutes=interval)
