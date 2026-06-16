"""Snapshot quality scoring and validation helpers."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from gex_core.features import safe_float
from gex_core.market_time import is_trader_session_active
from gex_core.spot_consensus import spot_disagreement_tolerance_pct
from gex_core.strike_filter import strikes_bracket_spot


def total_gex_tolerance_bn() -> float:
    try:
        return float(os.environ.get("GEX_TOTAL_GEX_TOLERANCE_BN", "0.05"))
    except (TypeError, ValueError):
        return 0.05


def max_data_lag_sec() -> float:
    try:
        return float(os.environ.get("GEX_MAX_DATA_LAG_SEC", "1200"))
    except (TypeError, ValueError):
        return 1200.0


def compute_data_lag_sec(uw_time: str | None, indexed_at: datetime | None = None) -> float | None:
    if not uw_time:
        return None
    try:
        observed = pd.Timestamp(uw_time)
        if observed.tzinfo is None:
            observed = observed.tz_localize("UTC")
        else:
            observed = observed.tz_convert("UTC")
        anchor = indexed_at or datetime.now(timezone.utc)
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        return max(0.0, (anchor - observed.to_pydatetime()).total_seconds())
    except (TypeError, ValueError):
        return None


def check_strike_completeness(gex_by_strike: pd.Series, spot: float) -> dict[str, Any]:
    strikes = pd.to_numeric(gex_by_strike.index, errors="coerce")
    if strikes.empty or spot <= 0:
        return {"ok": False, "reason": "empty_or_no_spot"}
    min_strike = float(strikes.min())
    max_strike = float(strikes.max())
    return {
        "ok": min_strike <= spot * 0.9 and max_strike >= spot * 1.05,
        "min_strike": min_strike,
        "max_strike": max_strike,
        "spot": spot,
        "brackets_spot": strikes_bracket_spot(gex_by_strike, spot),
    }


def regime_from_total_gex(total_gex: float) -> str:
    return "LONG gamma" if total_gex >= 0 else "SHORT gamma"


def regime_consistent(total_gex: float, cumulative_slope_at_spot: float | None) -> bool | None:
    if cumulative_slope_at_spot is None:
        return None
    label = regime_from_total_gex(total_gex)
    slope_regime = "LONG gamma" if cumulative_slope_at_spot >= 0 else "SHORT gamma"
    return label == slope_regime


def strike_profile_confidence(source: str | None) -> str:
    mapping = {
        "live_spot_exposures": "high",
        "spot_exposures": "high",
        "greek_exposure_atm": "medium",
        "greek_exposure_filtered": "medium",
        "eod_scaled": "low",
        "spot_exposures_misaligned": "low",
    }
    return mapping.get(str(source or ""), "medium")


def compute_quality_score(
    *,
    brackets_spot: bool,
    spot_disagreement_pct: float,
    total_gex_consistent: bool,
    data_lag_sec: float | None,
    strike_profile_source: str | None,
    strike_count: int,
    validation_ok: bool,
) -> float:
    score = 0.0
    if validation_ok:
        score += 0.15
    score += 0.30 if brackets_spot else 0.0
    if spot_disagreement_pct <= spot_disagreement_tolerance_pct():
        score += 0.20
    elif spot_disagreement_pct <= spot_disagreement_tolerance_pct() * 2:
        score += 0.10
    score += 0.20 if total_gex_consistent else 0.0
    if data_lag_sec is None:
        score += 0.10
    elif data_lag_sec <= max_data_lag_sec() * 0.5:
        score += 0.15
    elif data_lag_sec <= max_data_lag_sec():
        score += 0.08
    confidence = strike_profile_confidence(strike_profile_source)
    score += {"high": 0.15, "medium": 0.10, "low": 0.03}.get(confidence, 0.05)
    if strike_count >= 20:
        score += 0.05
    elif strike_count >= 10:
        score += 0.03
    return round(min(1.0, max(0.0, score)), 4)


def hard_reject_total_gex_mismatch() -> bool:
    return os.environ.get("GEX_HARD_REJECT_TOTAL_GEX_MISMATCH", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def should_hard_reject_total_gex_mismatch(*, mismatch: bool) -> bool:
    if not mismatch or not hard_reject_total_gex_mismatch():
        return False
    return is_trader_session_active()
