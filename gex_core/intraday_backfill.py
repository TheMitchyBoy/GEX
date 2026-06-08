"""Backfill minute-level and daily UW GEX exports for model training."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from gex_core.env_bootstrap import parse_env_minutes
from gex_core.exports import EXPORT_DIR, list_export_timestamps
from gex_core.extended_features import merge_extended_features
from gex_core.history import clear_history_cache, get_latest_ts
from gex_core.market_features import fetch_cross_asset_returns, fetch_vol_regime
from gex_core.refresh import recent_market_dates
from gex_core.snapshot_export import write_snapshot_export
from gex_core.tickers import is_supported_ticker

logger = logging.getLogger(__name__)

from gex_core.spot_exposure import RAW_SCALE as _RAW_GAMMA_SCALE
DEFAULT_BACKFILL_DAYS = int(os.environ.get("GEX_INTRADAY_BACKFILL_DAYS", "90"))
DEFAULT_BACKFILL_INTERVAL_MINUTES = parse_env_minutes("GEX_BACKFILL_INTERVAL_MINUTES", 10.0)
DEFAULT_LIVE_INTERVAL_MINUTES = parse_env_minutes("GEX_REFRESH_INTERVAL_MINUTES", 10.0)
DEFAULT_DAILY_BACKFILL_DAYS = int(os.environ.get("GEX_DAILY_BACKFILL_DAYS", "90"))


def uw_time_to_export_ts(value: datetime | str | pd.Timestamp) -> str:
    """Convert UW ``time`` to export suffix ``YYYY-MM-DD_HHMMSS`` (UTC)."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    else:
        ts = ts.tz_localize("UTC")
    return ts.strftime("%Y-%m-%d_%H%M%S")


def minute_row_total_gex_bn(row: pd.Series) -> float:
    """Estimate net GEX (Bn$ / %) from a 1-minute spot-exposure row."""
    for col in (
        "gamma_per_one_percent_move_oi",
        "gamma_per_one_percent_move_dir",
        "gamma_per_one_percent_move_vol",
    ):
        if col in row.index and pd.notna(row[col]):
            return float(row[col]) / _RAW_GAMMA_SCALE
    if "call_gamma_oi" in row.index and "put_gamma_oi" in row.index:
        call = pd.to_numeric(row["call_gamma_oi"], errors="coerce")
        put = pd.to_numeric(row["put_gamma_oi"], errors="coerce")
        if hasattr(call, "fillna"):
            call = float(call.fillna(0.0).sum())
            put = float(put.fillna(0.0).sum())
        else:
            call = float(call) if pd.notna(call) else 0.0
            put = float(put) if pd.notna(put) else 0.0
        return (call + put) / _RAW_GAMMA_SCALE
    return 0.0


def sample_intraday_rows(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Downsample 1-minute UW rows to one row per ``interval_minutes`` bucket."""
    if interval_minutes <= 1 or df.empty or "time" not in df.columns:
        return df
    sampled = df.dropna(subset=["time"]).copy()
    sampled["time"] = pd.to_datetime(sampled["time"], utc=True, errors="coerce")
    sampled = sampled.dropna(subset=["time"]).sort_values("time")
    if sampled.empty:
        return sampled
    sampled["_bucket"] = sampled["time"].dt.floor(f"{interval_minutes}min")
    return sampled.groupby("_bucket", as_index=False).last().drop(columns="_bucket")


def scale_strike_profile(strike: pd.Series, target_total_bn: float) -> pd.Series:
    """Scale an EOD strike profile to match a minute-level aggregate total."""
    if strike.empty:
        return strike
    current = float(strike.sum())
    if abs(current) < 1e-12:
        return strike
    return strike * (target_total_bn / current)


def _existing_timestamps(ticker: str, export_dir: Path) -> set[str]:
    return set(list_export_timestamps(ticker, export_dir))


def _build_summary(
    ticker: str,
    *,
    market_date: str,
    spot: float,
    total_gex_bn: float,
    uw_endpoint: str,
    granularity: str,
    uw_time: str | None = None,
    greek_df: pd.DataFrame | None = None,
    spot_df: pd.DataFrame | None = None,
    vol_regime: dict | None = None,
    cross_asset: dict | None = None,
) -> dict:
    summary: dict = {
        "ticker": ticker.upper(),
        "data_source": "unusual_whales",
        "source": "Unusual Whales API",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_date": market_date,
        "spot": float(spot),
        "spot_price": float(spot),
        "total_gex_bn_per_pct": float(total_gex_bn),
        "net_gamma_regime": "LONG gamma" if total_gex_bn >= 0 else "SHORT gamma",
        "uw_endpoint": uw_endpoint,
        "granularity": granularity,
    }
    if uw_time:
        summary["uw_time_utc"] = uw_time
    merge_extended_features(
        summary,
        greek_df=greek_df,
        spot_exposures_df=spot_df,
        market_date=market_date,
        vol_regime=vol_regime if vol_regime is not None else fetch_vol_regime(),
        cross_asset=cross_asset if cross_asset is not None else fetch_cross_asset_returns(),
    )
    return summary


def export_live_strike_snapshot(
    ticker: str,
    *,
    export_dir: Path | None = None,
    api_key: str | None = None,
    force: bool = False,
) -> str | None:
    """Fetch full strike GEX and write a minute-timestamped export."""
    from gex_core.uw_loader import fetch_uw_gex

    ticker = ticker.upper()
    export_dir = export_dir or EXPORT_DIR
    interval = DEFAULT_LIVE_INTERVAL_MINUTES
    if not force and _snapshot_fresh_within_interval(ticker, export_dir, interval_minutes=interval):
        logger.info("Skipping %s — snapshot already exists for current %.2g-min window", ticker, interval)
        return get_latest_ts(ticker, export_dir)

    spot, agg = fetch_uw_gex(ticker, api_key=api_key)
    market_date = agg.gex_by_strike.attrs.get("market_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    greek_df = agg.gex_by_strike.attrs.get("greek_exposure_df")
    spot_df = agg.gex_by_strike.attrs.get("spot_exposures_df")
    summary = _build_summary(
        ticker,
        market_date=market_date,
        spot=spot,
        total_gex_bn=agg.total_gex_bn,
        uw_endpoint="greek-exposure/strike",
        granularity=f"{interval}min",
        greek_df=greek_df if isinstance(greek_df, pd.DataFrame) else None,
        spot_df=spot_df if isinstance(spot_df, pd.DataFrame) else None,
    )
    summary["interval_minutes"] = interval
    ts = write_snapshot_export(
        ticker,
        gex_by_strike=agg.gex_by_strike,
        cumulative_gex=agg.cumulative_gex,
        gex_by_expiration=agg.gex_by_expiration,
        surface_data=agg.surface_data,
        greek_exposure_df=greek_df if isinstance(greek_df, pd.DataFrame) else None,
        summary=summary,
        export_dir=export_dir,
    )
    logger.info("Live minute snapshot saved for %s at %s", ticker, ts)
    return ts


def backfill_intraday_minutes(
    ticker: str,
    market_date: str,
    *,
    export_dir: Path | None = None,
    api_key: str | None = None,
    force: bool = False,
    interval_minutes: int | None = None,
) -> int:
    """Backfill UW spot-exposure rows for a single trading day."""
    from gex_core.uw_loader import fetch_uw_spot_exposures, fetch_uw_spot_exposures_intraday
    from gex_core.spot_exposure import spot_exposure_net_series

    ticker = ticker.upper()
    export_dir = export_dir or EXPORT_DIR
    if not is_supported_ticker(ticker):
        return 0

    interval_minutes = interval_minutes or DEFAULT_BACKFILL_INTERVAL_MINUTES
    minute_df = fetch_uw_spot_exposures_intraday(ticker, api_key=api_key, date=market_date)
    if minute_df.empty:
        logger.warning(
            "No intraday spot-exposures for %s on %s — falling back to EOD strike snapshot",
            ticker,
            market_date,
        )
        return 1 if backfill_daily_strike_snapshots(
            ticker,
            market_date,
            export_dir=export_dir,
            api_key=api_key,
            force=force,
        ) else 0
    minute_df = sample_intraday_rows(minute_df, interval_minutes)
    if minute_df.empty:
        logger.warning("No %d-min samples for %s on %s", interval_minutes, ticker, market_date)
        return 0

    try:
        spot_df = fetch_uw_spot_exposures(ticker, api_key=api_key, date=market_date)
        base_strike = spot_exposure_net_series(spot_df, "gamma")
        if base_strike.empty:
            logger.warning("Empty spot-exposures/strike profile for %s on %s", ticker, market_date)
            return 0
    except Exception:
        logger.exception("Could not load spot-exposure strike profile for %s on %s", ticker, market_date)
        return 0

    greek_exposure_df = pd.DataFrame()
    try:
        from gex_core.uw_loader import fetch_uw_greek_exposure

        greek_exposure_df = fetch_uw_greek_exposure(ticker, api_key=api_key, date=market_date)
    except Exception:
        logger.debug("Greek exposure unavailable for intraday backfill %s on %s", ticker, market_date)

    existing = _existing_timestamps(ticker, export_dir)
    vol_regime = fetch_vol_regime()
    cross_asset = fetch_cross_asset_returns()
    saved = 0
    for _, row in minute_df.iterrows():
        if pd.isna(row.get("time")):
            continue
        ts = uw_time_to_export_ts(row["time"])
        if not force and ts in existing:
            continue
        spot = float(row.get("price") or 0.0)
        total_gex_bn = minute_row_total_gex_bn(row)
        cumulative = base_strike.cumsum()
        summary = _build_summary(
            ticker,
            market_date=market_date,
            spot=spot,
            total_gex_bn=total_gex_bn,
            uw_endpoint="spot-exposures",
            granularity=f"{interval_minutes}min",
            uw_time=pd.Timestamp(row["time"]).isoformat(),
            vol_regime=vol_regime,
            cross_asset=cross_asset,
        )
        summary["interval_minutes"] = interval_minutes
        write_snapshot_export(
            ticker,
            gex_by_strike=base_strike,
            cumulative_gex=cumulative,
            greek_exposure_df=greek_exposure_df if not greek_exposure_df.empty else None,
            summary=summary,
            export_dir=export_dir,
            timestamp=ts,
        )
        existing.add(ts)
        saved += 1

    if saved:
        clear_history_cache()
    logger.info(
        "Saved %d %d-min snapshots for %s on %s",
        saved,
        interval_minutes,
        ticker,
        market_date,
    )
    return saved


def backfill_daily_strike_snapshots(
    ticker: str,
    market_date: str,
    *,
    export_dir: Path | None = None,
    api_key: str | None = None,
    force: bool = False,
    close_ts_suffix: str = "160000",
) -> bool:
    """Backfill one EOD strike-level snapshot for a historical date."""
    from gex_core.uw_loader import fetch_uw_gex

    ticker = ticker.upper()
    export_dir = export_dir or EXPORT_DIR
    ts = f"{market_date}_{close_ts_suffix}"
    if not force and ts in _existing_timestamps(ticker, export_dir):
        return True

    try:
        spot, agg = fetch_uw_gex(ticker, api_key=api_key, date=market_date)
    except Exception:
        logger.exception("Daily strike backfill failed for %s on %s", ticker, market_date)
        return False

    greek_df = agg.gex_by_strike.attrs.get("greek_exposure_df")
    summary = _build_summary(
        ticker,
        market_date=market_date,
        spot=spot,
        total_gex_bn=agg.total_gex_bn,
        uw_endpoint="greek-exposure/strike",
        granularity="eod",
        greek_df=greek_df if isinstance(greek_df, pd.DataFrame) else None,
    )
    write_snapshot_export(
        ticker,
        gex_by_strike=agg.gex_by_strike,
        cumulative_gex=agg.cumulative_gex,
        gex_by_expiration=agg.gex_by_expiration,
        surface_data=agg.surface_data,
        greek_exposure_df=greek_df if isinstance(greek_df, pd.DataFrame) else None,
        summary=summary,
        export_dir=export_dir,
        timestamp=ts,
    )
    return True


def backfill_recent_intraday(
    ticker: str,
    *,
    days: int | None = None,
    export_dir: Path | None = None,
    api_key: str | None = None,
    force: bool = False,
    interval_minutes: int | None = None,
) -> dict[str, int]:
    """Backfill intraday exports for recent weekdays."""
    days = days if days is not None else DEFAULT_BACKFILL_DAYS
    results: dict[str, int] = {}
    for market_date in recent_market_dates(days=days):
        results[market_date] = backfill_intraday_minutes(
            ticker,
            market_date,
            export_dir=export_dir,
            api_key=api_key,
            force=force,
            interval_minutes=interval_minutes,
        )
    return results


def backfill_recent_daily(
    ticker: str,
    *,
    days: int | None = None,
    export_dir: Path | None = None,
    api_key: str | None = None,
    force: bool = False,
) -> dict[str, bool]:
    """Backfill EOD strike snapshots for recent weekdays."""
    days = days if days is not None else DEFAULT_DAILY_BACKFILL_DAYS
    results: dict[str, bool] = {}
    for market_date in recent_market_dates(days=days):
        results[market_date] = backfill_daily_strike_snapshots(
            ticker,
            market_date,
            export_dir=export_dir,
            api_key=api_key,
            force=force,
        )
    return results


def _snapshot_fresh_within_interval(
    ticker: str,
    export_dir: Path,
    *,
    interval_minutes: float,
) -> bool:
    latest = get_latest_ts(ticker, export_dir)
    if not latest:
        return False
    from gex_core.exports import parse_timestamp

    latest_ts = parse_timestamp(latest)
    age = datetime.now() - latest_ts
    return age <= timedelta(minutes=interval_minutes)
