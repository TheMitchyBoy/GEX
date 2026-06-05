"""Backfill minute-level and daily UW GEX exports for model training."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from gex_core.exports import EXPORT_DIR, list_export_timestamps
from gex_core.extended_features import merge_extended_features
from gex_core.history import clear_history_cache, get_latest_ts
from gex_core.market_features import fetch_cross_asset_returns, fetch_vol_regime
from gex_core.refresh import recent_market_dates
from gex_core.snapshot_export import write_snapshot_export
from gex_core.tickers import is_supported_ticker

logger = logging.getLogger(__name__)

_RAW_GAMMA_SCALE = 1e9  # spot-exposures aggregate fields are raw dollars per 1%


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
    if not force and _snapshot_fresh_this_minute(ticker, export_dir):
        logger.info("Skipping %s — snapshot already exists for current minute", ticker)
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
        granularity="minute",
        greek_df=greek_df if isinstance(greek_df, pd.DataFrame) else None,
        spot_df=spot_df if isinstance(spot_df, pd.DataFrame) else None,
    )
    ts = write_snapshot_export(
        ticker,
        gex_by_strike=agg.gex_by_strike,
        cumulative_gex=agg.cumulative_gex,
        gex_by_expiration=agg.gex_by_expiration,
        surface_data=agg.surface_data,
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
) -> int:
    """Backfill 1-minute UW spot-exposure rows for a single trading day."""
    from gex_core.uw_loader import fetch_uw_greek_exposure, fetch_uw_spot_exposures_intraday

    ticker = ticker.upper()
    export_dir = export_dir or EXPORT_DIR
    if not is_supported_ticker(ticker):
        return 0

    minute_df = fetch_uw_spot_exposures_intraday(ticker, api_key=api_key, date=market_date)
    if minute_df.empty:
        logger.warning("No intraday spot-exposures for %s on %s", ticker, market_date)
        return 0

    try:
        strike_df = fetch_uw_greek_exposure(ticker, api_key=api_key, date=market_date)
        base_strike = pd.Series(
            strike_df["net_gex"].values,
            index=strike_df["strike"].values,
            dtype=float,
        ).sort_index()
    except Exception:
        logger.exception("Could not load strike profile for %s on %s", ticker, market_date)
        return 0

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
        scaled_strike = scale_strike_profile(base_strike, total_gex_bn)
        cumulative = scaled_strike.cumsum()
        summary = _build_summary(
            ticker,
            market_date=market_date,
            spot=spot,
            total_gex_bn=total_gex_bn,
            uw_endpoint="spot-exposures",
            granularity="minute",
            uw_time=pd.Timestamp(row["time"]).isoformat(),
            vol_regime=vol_regime,
            cross_asset=cross_asset,
        )
        write_snapshot_export(
            ticker,
            gex_by_strike=scaled_strike,
            cumulative_gex=cumulative,
            summary=summary,
            export_dir=export_dir,
            timestamp=ts,
        )
        existing.add(ts)
        saved += 1

    if saved:
        clear_history_cache()
    logger.info("Saved %d minute snapshots for %s on %s", saved, ticker, market_date)
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
        summary=summary,
        export_dir=export_dir,
        timestamp=ts,
    )
    return True


def backfill_recent_intraday(
    ticker: str,
    *,
    days: int = 7,
    export_dir: Path | None = None,
    api_key: str | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Backfill minute-level exports for recent weekdays."""
    results: dict[str, int] = {}
    for market_date in recent_market_dates(days=days):
        results[market_date] = backfill_intraday_minutes(
            ticker,
            market_date,
            export_dir=export_dir,
            api_key=api_key,
            force=force,
        )
    return results


def backfill_recent_daily(
    ticker: str,
    *,
    days: int = 30,
    export_dir: Path | None = None,
    api_key: str | None = None,
    force: bool = False,
) -> dict[str, bool]:
    """Backfill EOD strike snapshots for recent weekdays."""
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


def _snapshot_fresh_this_minute(ticker: str, export_dir: Path) -> bool:
    latest = get_latest_ts(ticker, export_dir)
    if not latest:
        return False
    from gex_core.exports import parse_timestamp

    latest_minute = parse_timestamp(latest).replace(second=0, microsecond=0)
    now_minute = datetime.now().replace(second=0, microsecond=0)
    return latest_minute >= now_minute
