"""Snapshot validation, dedup, and derived feature computation before Postgres write."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from gex_core.exports import parse_timestamp
from gex_core.features import (
    extract_surface_vector,
    resolve_gamma_flip,
    safe_float,
    select_atm_strike_series,
    term_structure_breakdown,
    top_strike_concentration,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    ok: bool
    status: str
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "status": self.status, "issues": self.issues, "warnings": self.warnings}


@dataclass
class PreparedSnapshot:
    ts: str
    snapshot_at: datetime
    summary: dict[str, Any]
    features: dict[str, Any]
    validation: ValidationResult
    skipped_duplicate: bool = False
    prior_ts: str | None = None
    atm_window_pct: float = 0.03
    strike_profile_hash: str | None = None
    data_quality: dict[str, Any] | None = None


def _atm_window_pct() -> float:
    try:
        return float(os.environ.get("GEX_ATM_STRIKE_WINDOW_PCT", "0.03"))
    except (TypeError, ValueError):
        return 0.03


def strike_profile_hash(strike: pd.Series) -> str:
    """Stable hash of a strike GEX profile for dedup."""
    if strike is None or strike.empty:
        return hashlib.sha256(b"empty").hexdigest()[:16]
    cleaned = pd.Series(pd.to_numeric(strike, errors="coerce"), index=pd.to_numeric(strike.index, errors="coerce"))
    cleaned = cleaned.dropna().sort_index()
    if cleaned.empty:
        return hashlib.sha256(b"empty").hexdigest()[:16]
    payload = json.dumps(
        {"strikes": [float(x) for x in cleaned.index], "gex": [float(x) for x in cleaned.values]},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def strikes_equal(left: pd.Series, right: pd.Series) -> bool:
    if left is None or right is None:
        return False
    if left.empty and right.empty:
        return True
    if left.empty or right.empty:
        return False
    return strike_profile_hash(left) == strike_profile_hash(right)


def export_ts_from_uw_time(
    uw_time: datetime | str | pd.Timestamp | None,
    *,
    interval_minutes: float,
    fallback: datetime | None = None,
) -> str:
    """Canonical export key from UW observation time, bucketed to refresh interval."""
    from gex_core.intraday_backfill import uw_time_to_export_ts

    if uw_time is not None and not (isinstance(uw_time, float) and pd.isna(uw_time)):
        ts = pd.Timestamp(uw_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if interval_minutes > 0:
            ts = ts.floor(f"{int(max(1, round(interval_minutes)))}min")
        return uw_time_to_export_ts(ts)
    anchor = fallback or datetime.now(timezone.utc)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    if interval_minutes > 0:
        ts = pd.Timestamp(anchor).tz_convert("UTC").floor(f"{int(max(1, round(interval_minutes)))}min")
        return uw_time_to_export_ts(ts)
    return anchor.strftime("%Y-%m-%d_%H%M%S")


def parse_snapshot_at(ts: str) -> datetime:
    return parse_timestamp(ts).replace(tzinfo=timezone.utc)


def build_spot_data_quality_report(
    *,
    strike_count: int,
    spot: float,
    total_gex_bn: float,
    strike_sum_bn: float,
) -> dict[str, Any]:
    removed: dict[str, int] = {}
    if strike_count <= 0:
        removed["empty_profile"] = 1
    if spot <= 0:
        removed["invalid_spot"] = 1
    tolerance = float(os.environ.get("GEX_TOTAL_GEX_TOLERANCE_BN", "0.05"))
    if abs(total_gex_bn - strike_sum_bn) > tolerance:
        removed["total_gex_mismatch"] = 1
    rows_out = 0 if removed else strike_count
    return {
        "rows_in": strike_count,
        "rows_out": rows_out,
        "filters_enabled": True,
        "removed": removed,
        "source": "spot_exposures/strike",
    }


def derive_snapshot_features(
    *,
    ticker: str,
    ts: str,
    gex_by_strike: pd.Series,
    cumulative_gex: pd.Series,
    gex_by_expiration: pd.Series | None,
    summary: dict[str, Any],
    prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strike = pd.Series(gex_by_strike, dtype=float).sort_index()
    spot = safe_float(summary.get("spot") or summary.get("spot_price"), 0.0)
    total_gex = safe_float(summary.get("total_gex_bn_per_pct"), float(strike.sum()) if len(strike) else 0.0)
    call_wall = float(strike.idxmax()) if len(strike) else None
    put_wall = float(strike.idxmin()) if len(strike) else None
    positive = strike[strike > 0]
    pos_gamma_peak = float(positive.idxmax()) if len(positive) else None
    gamma_flip = resolve_gamma_flip(
        spot=spot if spot > 0 else None,
        gex_by_strike=strike if len(strike) else None,
        cumulative_gex=pd.Series(cumulative_gex, dtype=float).sort_index() if len(cumulative_gex) else None,
    )
    if gamma_flip is None:
        gamma_flip = safe_float(summary.get("gamma_flip"), 0.0) or None
    flip_distance_pct = ((gamma_flip - spot) / spot) if gamma_flip and spot > 0 else None
    wall_spread = (call_wall - put_wall) if call_wall is not None and put_wall is not None else None
    term = term_structure_breakdown(
        gex_by_expiration if gex_by_expiration is not None else pd.Series(dtype=float),
        snapshot_date=pd.Timestamp(parse_snapshot_at(ts)),
    )
    surface_vector = extract_surface_vector(strike, spot).tolist()
    prior_ts = prior.get("ts") if prior else None
    prior_spot = safe_float(prior.get("spot"), 0.0) if prior else 0.0
    prior_gex = safe_float(prior.get("total_gex"), 0.0) if prior else 0.0
    prior_regime = str(prior.get("regime") or "") if prior else ""
    delta_spot = (spot - prior_spot) if prior and prior_spot > 0 and spot > 0 else None
    spot_return = (delta_spot / prior_spot) if delta_spot is not None and prior_spot > 0 else None
    delta_gex = (total_gex - prior_gex) if prior else None
    regime = str(summary.get("net_gamma_regime") or ("LONG gamma" if total_gex >= 0 else "SHORT gamma"))
    regime_changed = bool(prior_regime and prior_regime != regime)
    return {
        "ticker": ticker.upper(),
        "ts": ts,
        "prior_ts": prior_ts,
        "snapshot_at": parse_snapshot_at(ts).isoformat(),
        "gamma_flip": gamma_flip,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "pos_gamma_peak_strike": pos_gamma_peak,
        "flip_distance_pct": flip_distance_pct,
        "wall_spread": wall_spread,
        "gex_concentration": top_strike_concentration(strike) if len(strike) else 0.0,
        "near_term_ratio": term.get("near_term_ratio"),
        "zero_dte_ratio": term.get("zero_dte_ratio"),
        "term_curvature": term.get("term_curvature"),
        "expiration_count": term.get("expiration_count"),
        "front_term_ratio": term.get("front_term_ratio"),
        "back_term_ratio": term.get("back_term_ratio"),
        "delta_gex": delta_gex,
        "delta_spot": delta_spot,
        "spot_return": spot_return,
        "regime_changed": regime_changed,
        "surface_vector": surface_vector,
        "strike_profile_hash": strike_profile_hash(strike),
        "strike_count": int(len(strike)),
    }


def enrich_summary_with_derived(
    summary: dict[str, Any],
    features: dict[str, Any],
    *,
    data_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(summary)
    for key in (
        "gamma_flip",
        "call_wall",
        "put_wall",
        "pos_gamma_peak_strike",
        "flip_distance_pct",
        "wall_spread",
        "gex_concentration",
        "near_term_ratio",
        "zero_dte_ratio",
        "term_curvature",
        "expiration_count",
        "front_term_ratio",
        "back_term_ratio",
    ):
        value = features.get(key)
        if value is not None:
            out[key] = value
    if data_quality is not None:
        out["data_quality"] = data_quality
    if features.get("prior_ts"):
        out["prior_ts"] = features["prior_ts"]
    if features.get("delta_gex") is not None:
        out["delta_gex_bn"] = features["delta_gex"]
    if features.get("spot_return") is not None:
        out["spot_return"] = features["spot_return"]
    return out


def validate_snapshot(
    *,
    ticker: str,
    ts: str,
    gex_by_strike: pd.Series,
    summary: dict[str, Any],
    prior: dict[str, Any] | None = None,
) -> ValidationResult:
    issues: list[str] = []
    warnings: list[str] = []
    strike = pd.Series(gex_by_strike, dtype=float)
    spot = safe_float(summary.get("spot") or summary.get("spot_price"), 0.0)
    total_gex = safe_float(summary.get("total_gex_bn_per_pct"), float(strike.sum()) if len(strike) else 0.0)
    strike_sum = float(strike.sum()) if len(strike) else 0.0

    if spot <= 0:
        issues.append("spot<=0")
    if not np.isfinite(total_gex):
        issues.append("total_gex_nan")
    if strike.empty:
        issues.append("empty_strike_profile")
    tolerance = float(os.environ.get("GEX_TOTAL_GEX_TOLERANCE_BN", "0.05"))
    if len(strike) and abs(total_gex - strike_sum) > tolerance:
        warnings.append("total_gex_strike_sum_mismatch")

    if prior:
        prior_count = int(prior.get("strike_count") or 0)
        if prior_count > 0 and len(strike) < prior_count * 0.5:
            warnings.append("strike_count_drop")

    min_strikes = int(os.environ.get("GEX_MIN_STRIKE_COUNT", "5"))
    if len(strike) < min_strikes:
        issues.append("too_few_strikes")

    if issues:
        return ValidationResult(ok=False, status="rejected", issues=issues, warnings=warnings)
    status = "ok_with_warnings" if warnings else "ok"
    return ValidationResult(ok=True, status=status, issues=issues, warnings=warnings)


def atm_strike_series(gex_by_strike: pd.Series, spot: float, *, window_pct: float | None = None) -> pd.Series:
    window_pct = _atm_window_pct() if window_pct is None else window_pct
    return select_atm_strike_series(gex_by_strike, spot, window_pct=window_pct, min_strikes=3)


def fetch_prior_snapshot(ticker: str) -> dict[str, Any] | None:
    from gex_core.db import use_postgres

    if not use_postgres():
        return None
    try:
        import psycopg

        from gex_core.db import database_url, ensure_postgres_schema

        ensure_postgres_schema()
        with psycopg.connect(database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT s.ts, s.spot, s.total_gex, s.regime, f.strike_profile_hash, f.strike_count
                    FROM snapshots s
                    LEFT JOIN snapshot_features f
                      ON f.ticker = s.ticker AND f.ts = s.ts
                    WHERE s.ticker = %s
                    ORDER BY s.ts DESC
                    LIMIT 1
                    """,
                    (ticker.upper(),),
                )
                row = cur.fetchone()
                if not row:
                    return None
                cur.execute(
                    """
                    SELECT strike, gex_bn_per_pct
                    FROM snapshot_strikes
                    WHERE ticker = %s AND ts = %s
                    ORDER BY strike
                    """,
                    (ticker.upper(), str(row[0])),
                )
                strikes = cur.fetchall()
        strike_series = pd.Series(
            {float(s): float(g) for s, g in strikes},
            dtype=float,
        )
        return {
            "ts": str(row[0]),
            "spot": row[1],
            "total_gex": row[2],
            "regime": row[3],
            "strike_profile_hash": row[4],
            "strike_count": row[5] or len(strike_series),
            "strike": strike_series,
        }
    except Exception:
        logger.debug("fetch_prior_snapshot failed for %s", ticker, exc_info=True)
        return None


def prepare_snapshot_for_storage(
    ticker: str,
    *,
    gex_by_strike: pd.Series,
    cumulative_gex: pd.Series,
    gex_by_expiration: pd.Series | None,
    summary: dict[str, Any],
    timestamp: str | None = None,
    uw_time: datetime | str | pd.Timestamp | None = None,
    interval_minutes: float | None = None,
    force: bool = False,
) -> PreparedSnapshot:
    from gex_core.env_bootstrap import parse_env_minutes
    from gex_core.refresh import DEFAULT_REFRESH_MINUTES

    interval = interval_minutes if interval_minutes is not None else parse_env_minutes(
        "GEX_REFRESH_INTERVAL_MINUTES", DEFAULT_REFRESH_MINUTES
    )
    ts = timestamp or export_ts_from_uw_time(uw_time, interval_minutes=interval)
    snapshot_at = parse_snapshot_at(ts)
    prior = fetch_prior_snapshot(ticker)
    strike_sum = float(pd.Series(gex_by_strike, dtype=float).sum())
    spot = safe_float(summary.get("spot") or summary.get("spot_price"), 0.0)
    total_gex = safe_float(summary.get("total_gex_bn_per_pct"), strike_sum)
    summary = dict(summary)
    summary.setdefault("total_gex_bn_per_pct", total_gex)
    data_quality = build_spot_data_quality_report(
        strike_count=len(gex_by_strike),
        spot=spot,
        total_gex_bn=total_gex,
        strike_sum_bn=strike_sum,
    )
    features = derive_snapshot_features(
        ticker=ticker,
        ts=ts,
        gex_by_strike=gex_by_strike,
        cumulative_gex=cumulative_gex,
        gex_by_expiration=gex_by_expiration,
        summary=summary,
        prior=prior,
    )
    summary = enrich_summary_with_derived(summary, features, data_quality=data_quality)
    validation = validate_snapshot(
        ticker=ticker,
        ts=ts,
        gex_by_strike=gex_by_strike,
        summary=summary,
        prior=prior,
    )
    skipped_duplicate = False
    if (
        not force
        and prior
        and prior.get("strike") is not None
        and strikes_equal(pd.Series(gex_by_strike, dtype=float), prior["strike"])
        and os.environ.get("GEX_SKIP_DUPLICATE_SNAPSHOTS", "1").strip().lower() in {"1", "true", "yes", "on"}
    ):
        skipped_duplicate = True
        validation = ValidationResult(ok=True, status="skipped_duplicate", warnings=["identical_strike_profile"])
    return PreparedSnapshot(
        ts=ts,
        snapshot_at=snapshot_at,
        summary=summary,
        features=features,
        validation=validation,
        skipped_duplicate=skipped_duplicate,
        prior_ts=features.get("prior_ts"),
        atm_window_pct=_atm_window_pct(),
        strike_profile_hash=features.get("strike_profile_hash"),
        data_quality=data_quality,
    )
