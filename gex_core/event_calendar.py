"""Macro and options calendar flags for prediction context."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from gex_core.exports import parse_timestamp

# Approximate 2026 US macro/options calendar (extend annually).
_FOMC_MEETING_WEEKS = {
    date(2026, 1, 26),
    date(2026, 3, 16),
    date(2026, 4, 27),
    date(2026, 6, 16),
    date(2026, 7, 28),
    date(2026, 9, 15),
    date(2026, 11, 4),
    date(2026, 12, 15),
}
_CPI_RELEASE_DAYS = {
    date(2026, 1, 14),
    date(2026, 2, 11),
    date(2026, 3, 11),
    date(2026, 4, 10),
    date(2026, 5, 13),
    date(2026, 6, 10),
    date(2026, 7, 14),
    date(2026, 8, 12),
    date(2026, 9, 10),
    date(2026, 10, 14),
    date(2026, 11, 12),
    date(2026, 12, 10),
}
_NFP_RELEASE_DAYS = {
    date(2026, 1, 9),
    date(2026, 2, 6),
    date(2026, 3, 6),
    date(2026, 4, 3),
    date(2026, 5, 8),
    date(2026, 6, 5),
    date(2026, 7, 2),
    date(2026, 8, 7),
    date(2026, 9, 4),
    date(2026, 10, 2),
    date(2026, 11, 6),
    date(2026, 12, 4),
}
_MONTHLY_OPEX = {date(2026, m, 20) for m in range(1, 13)}
_QUAD_WITCHING = {date(2026, 3, 20), date(2026, 6, 19), date(2026, 9, 18), date(2026, 12, 18)}


def _as_date(value: date | datetime | str | pd.Timestamp | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, str):
        try:
            return parse_timestamp(value).date()
        except ValueError:
            return pd.Timestamp(value).date()
    return None


def _near(target: date, events: set[date], *, days: int) -> bool:
    return any(abs((target - event).days) <= days for event in events)


def event_calendar_features(
    when: date | datetime | str | pd.Timestamp | None = None,
) -> dict[str, float]:
    """Return calendar risk flags for a snapshot date."""
    day = _as_date(when) or datetime.utcnow().date()
    is_fomc_week = 1.0 if _near(day, _FOMC_MEETING_WEEKS, days=3) else 0.0
    is_cpi_day = 1.0 if _near(day, _CPI_RELEASE_DAYS, days=0) else 0.0
    is_nfp_day = 1.0 if _near(day, _NFP_RELEASE_DAYS, days=0) else 0.0
    is_opex_week = 1.0 if _near(day, _MONTHLY_OPEX, days=2) else 0.0
    is_quad_witching = 1.0 if _near(day, _QUAD_WITCHING, days=1) else 0.0
    event_risk_score = min(
        1.0,
        0.35 * is_fomc_week
        + 0.25 * is_cpi_day
        + 0.15 * is_nfp_day
        + 0.15 * is_opex_week
        + 0.25 * is_quad_witching,
    )
    return {
        "is_fomc_week": is_fomc_week,
        "is_cpi_day": is_cpi_day,
        "is_nfp_day": is_nfp_day,
        "is_opex_week": is_opex_week,
        "is_quad_witching": is_quad_witching,
        "event_risk_score": event_risk_score,
    }
