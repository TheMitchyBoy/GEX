"""Fast Periscope data path: UW API for today/same-day, SQLite index for history."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from gex_core.charts import safe_float
from gex_core.env_bootstrap import uw_api_configured, uw_api_key
from gex_core.features import enrich_snapshot_metrics, estimate_gamma_flip
from gex_core.history import load_snapshot_at_ts
from gex_core.market_time import (
    market_now_export_ts,
    market_today,
    ts_display_label,
    ts_market_date,
)
from gex_core.intraday_backfill import (
    minute_row_total_gex_bn,
    sample_intraday_rows,
    uw_time_to_export_ts,
)
from gex_core.spot_exposure import spot_exposure_net_series
from gex_core.storage import (
    list_indexed_dates,
    list_indexed_timestamps_before_date,
    list_indexed_timestamps_for_date,
)
from gex_core.tickers import PRIMARY_TICKER

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = int(os.environ.get("GEX_BACKFILL_INTERVAL_MINUTES", "10"))
_CACHE_TTL = int(
    os.environ.get(
        "GEX_PERISCOPE_API_CACHE_TTL_SECONDS",
        str(max(30, int(os.environ.get("GEX_REFRESH_INTERVAL_MINUTES", "10")) * 60)),
    )
)

_day_cache: dict[tuple[str, str], "IntradayDayCache"] = {}
_cache_lock = threading.Lock()


@dataclass
class IntradayDayCache:
    """In-memory same-day slices built from UW spot-exposures API."""

    market_date: str
    timestamps: list[str] = field(default_factory=list)
    snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    price_points: list[dict[str, Any]] = field(default_factory=list)
    fetched_at: float = 0.0
    source: str = "uw_api"


def _cache_fresh(entry: IntradayDayCache | None) -> bool:
    return entry is not None and (time.monotonic() - entry.fetched_at) < _CACHE_TTL


def _snapshot_from_strike(
    ticker: str,
    ts: str,
    *,
    strike: pd.Series,
    spot: float,
    total_gex_bn: float,
    data_source: str = "uw_api",
    spot_exposures_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    cumulative = strike.cumsum()
    call_wall = float(strike.idxmax()) if len(strike) else None
    put_wall = float(strike.idxmin()) if len(strike) else None
    metrics: dict[str, Any] = {
        "ts": ts,
        "ts_label": ts_display_label(ts),
        "market_date": ts_market_date(ts),
        "ticker": ticker.upper(),
        "strike": strike,
        "cumulative": cumulative,
        "surface_df": pd.DataFrame(),
        "total_gex": float(total_gex_bn),
        "pos_gex": float(strike[strike > 0].sum()),
        "neg_gex": float(strike[strike < 0].sum()),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": estimate_gamma_flip(cumulative),
        "regime": "LONG gamma" if total_gex_bn >= 0 else "SHORT gamma",
        "data_source": data_source,
        "spot": float(spot),
        "uw_endpoint": "spot-exposures/strike",
    }
    if spot_exposures_df is not None and not spot_exposures_df.empty:
        metrics["spot_exposures_df"] = spot_exposures_df
    return enrich_snapshot_metrics(metrics)


def snapshot_from_uw_entry(ticker: str, uw_entry: dict[str, Any], ts: str | None = None) -> dict[str, Any]:
    """Build a Periscope snapshot from live UW spot-exposures (matches UW Periscope)."""
    agg = uw_entry["agg"]
    spot_df = agg.gex_by_strike.attrs.get("spot_exposures_df")
    spot = safe_float(uw_entry.get("spot"), 0.0)
    if isinstance(spot_df, pd.DataFrame) and not spot_df.empty:
        strike = spot_exposure_net_series(spot_df, "gamma")
        if spot <= 0 and "price" in spot_df.columns:
            spot = safe_float(spot_df["price"].dropna().iloc[0], 0.0)
    else:
        spot_df = None
        strike = pd.Series(agg.gex_by_strike, dtype=float).sort_index()
    total_gex = safe_float(
        uw_entry.get("spot_gamma_bn"),
        safe_float(agg.total_gex_bn, float(strike.sum())),
    )
    active_ts = ts or market_now_export_ts()
    return _snapshot_from_strike(
        ticker,
        active_ts,
        strike=strike,
        spot=spot,
        total_gex_bn=total_gex,
        data_source="unusual_whales",
        spot_exposures_df=spot_df if isinstance(spot_df, pd.DataFrame) else None,
    )


def fetch_intraday_day_cache(
    ticker: str,
    market_date: str,
    *,
    api_key: str | None = None,
    interval_minutes: int | None = None,
    force: bool = False,
) -> IntradayDayCache | None:
    """Fetch and cache all intraday slices for one trading day via UW API."""
    ticker = ticker.upper()
    market_date = market_date[:10]
    cache_key = (ticker, market_date)
    interval_minutes = interval_minutes or _DEFAULT_INTERVAL

    with _cache_lock:
        cached = _day_cache.get(cache_key)
        if not force and _cache_fresh(cached):
            return cached

    api_key = api_key or uw_api_key()
    if not api_key:
        return None

    from gex_core.uw_loader import fetch_uw_spot_exposures, fetch_uw_spot_exposures_intraday

    try:
        minute_df = fetch_uw_spot_exposures_intraday(ticker, api_key=api_key, date=market_date)
        if minute_df.empty:
            return cached if cached else None

        spot_df = fetch_uw_spot_exposures(ticker, api_key=api_key, date=market_date)
        base_strike = spot_exposure_net_series(spot_df, "gamma")
        if base_strike.empty:
            return cached if cached else None
    except Exception:
        logger.exception("UW intraday fetch failed for %s on %s", ticker, market_date)
        return cached if cached else None

    sampled = sample_intraday_rows(minute_df, interval_minutes)
    timestamps: list[str] = []
    snapshots: dict[str, dict[str, Any]] = {}
    price_points: list[dict[str, Any]] = []

    for _, row in sampled.iterrows():
        if pd.isna(row.get("time")):
            continue
        ts = uw_time_to_export_ts(row["time"])
        if ts_market_date(ts) != market_date:
            continue
        spot = safe_float(row.get("price"), 0.0)
        total_gex_bn = minute_row_total_gex_bn(row)
        snapshots[ts] = _snapshot_from_strike(
            ticker,
            ts,
            strike=base_strike,
            spot=spot,
            total_gex_bn=total_gex_bn,
            spot_exposures_df=spot_df,
        )
        timestamps.append(ts)
        if spot > 0:
            price_points.append(
                {
                    "ts": ts_display_label(ts),
                    "close": spot,
                    "time": pd.Timestamp(row["time"]).isoformat(),
                }
            )

    if not timestamps:
        return cached if cached else None

    entry = IntradayDayCache(
        market_date=market_date,
        timestamps=sorted(timestamps),
        snapshots=snapshots,
        price_points=price_points,
        fetched_at=time.monotonic(),
    )
    with _cache_lock:
        _day_cache[cache_key] = entry
    return entry


def list_api_intraday_timestamps(
    ticker: str,
    market_date: str,
    *,
    api_key: str | None = None,
) -> list[str]:
    cache = fetch_intraday_day_cache(ticker, market_date, api_key=api_key)
    return list(cache.timestamps) if cache else []


def list_periscope_timestamps(
    ticker: str = PRIMARY_TICKER,
    *,
    api_key: str | None = None,
    today: str | None = None,
) -> list[str]:
    """
    Timestamp catalog for Periscope without scanning thousands of CSV files.

    Historical days come from the SQLite index; today uses UW intraday API.
    """
    ticker = ticker.upper()
    today = today or market_today()
    historical = list_indexed_timestamps_before_date(ticker, today)

    if uw_api_configured() or api_key:
        api_today = list_api_intraday_timestamps(ticker, today, api_key=api_key)
        if api_today:
            return historical + api_today

    today_indexed = list_indexed_timestamps_for_date(ticker, today)
    if today_indexed:
        return historical + today_indexed
    return historical


def list_periscope_dates(
    ticker: str = PRIMARY_TICKER,
    *,
    api_key: str | None = None,
    today: str | None = None,
) -> list[str]:
    """Available trading days for the calendar picker."""
    today = today or market_today()
    dates = list_indexed_dates(ticker)
    if today not in dates and (uw_api_configured() or api_key):
        api_ts = list_api_intraday_timestamps(ticker, today, api_key=api_key)
        if api_ts:
            dates = sorted(set(dates) | {today})
    return dates


def should_use_api_for_date(market_date: str | None, *, api_key: str | None = None) -> bool:
    if not market_date or not (uw_api_configured() or api_key):
        return False
    return market_date[:10] >= market_today()


def load_periscope_snapshot(
    ticker: str,
    ts: str | None,
    *,
    api_key: str | None = None,
    uw_entry: dict[str, Any] | None = None,
    market_date: str | None = None,
) -> dict[str, Any] | None:
    """Load one snapshot via API (same-day) or a single indexed CSV."""
    ticker = ticker.upper()
    if not ts:
        if uw_entry and uw_entry.get("agg") is not None:
            return snapshot_from_uw_entry(ticker, uw_entry)
        return None

    day = (market_date or ts_market_date(ts))[:10]
    if should_use_api_for_date(day, api_key=api_key):
        cache = fetch_intraday_day_cache(ticker, day, api_key=api_key)
        if cache and ts in cache.snapshots:
            return cache.snapshots[ts]
        if uw_entry and uw_entry.get("agg") is not None and cache and cache.timestamps and ts == cache.timestamps[-1]:
            return snapshot_from_uw_entry(ticker, uw_entry, ts=ts)

    return load_snapshot_at_ts(ticker, ts)


def periscope_price_points(
    ticker: str,
    *,
    market_date: str | None = None,
    api_key: str | None = None,
    fallback_history: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Intraday price series for the price chart — API first for same-day."""
    day = (market_date or market_today())[:10]
    if should_use_api_for_date(day, api_key=api_key):
        cache = fetch_intraday_day_cache(ticker, day, api_key=api_key)
        if cache and cache.price_points:
            return cache.price_points

    if not fallback_history:
        return []
    points = []
    for row in fallback_history:
        spot = safe_float(row.get("spot"), 0.0)
        if spot > 0:
            points.append({"ts": row.get("ts_label"), "close": spot})
    return points


def clear_periscope_api_cache() -> None:
    with _cache_lock:
        _day_cache.clear()
