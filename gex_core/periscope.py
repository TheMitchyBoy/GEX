"""Periscope-style market maker exposure dashboard data assembly."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gex_core.charts import safe_float
from gex_core.env_bootstrap import uw_api_key
from gex_core.features import estimate_gamma_flip
from gex_core.market_time import (
    market_today,
    ts_display_label,
    ts_market_date,
    ts_market_time_label,
)
from gex_core.periscope_api import (
    list_periscope_dates,
    list_periscope_timestamps,
    load_periscope_snapshot,
    periscope_price_points,
    should_use_api_for_date,
)
from gex_core.storage import list_indexed_timestamps_for_date
from gex_core.tickers import PRIMARY_TICKER

EXPOSURE_TYPES = ("gamma", "vanna", "charm")


def group_timestamps_by_date(timestamps: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for ts in sorted(timestamps):
        grouped.setdefault(ts_market_date(ts), []).append(ts)
    return grouped


def available_dates(timestamps: list[str], *, ticker: str = PRIMARY_TICKER) -> list[str]:
    indexed = list_periscope_dates(ticker, api_key=uw_api_key())
    if indexed:
        return indexed
    return sorted(group_timestamps_by_date(timestamps).keys())


def slices_for_date(
    timestamps: list[str],
    date: str,
    *,
    ticker: str = PRIMARY_TICKER,
) -> list[str]:
    """Slices for a trading session date (market timezone, not UTC key prefix)."""
    market_date = date[:10]
    from_index = list_indexed_timestamps_for_date(ticker, market_date)
    if from_index:
        return from_index
    return group_timestamps_by_date(timestamps).get(market_date, [])


def resolve_selected_timestamp(
    timestamps: list[str],
    *,
    ts: str | None = None,
    date: str | None = None,
) -> str | None:
    """Pick the active slice from explicit ts, a calendar day, or latest."""
    if ts:
        return ts
    if not timestamps:
        return None
    if date:
        day_slices = slices_for_date(timestamps, date, ticker=PRIMARY_TICKER)
        if day_slices:
            return day_slices[-1]
    return timestamps[-1]


def build_slice_options(
    timestamps: list[str],
    date: str,
    *,
    ticker: str = PRIMARY_TICKER,
) -> list[dict[str, str]]:
    """Slice dropdown rows for one trading day."""
    return [
        {
            "ts": ts,
            "time": ts_market_time_label(ts),
            "label": ts_display_label(ts),
        }
        for ts in slices_for_date(timestamps, date, ticker=ticker)
    ]


def build_timeline_navigation(
    timestamps: list[str],
    selected_ts: str | None,
    *,
    ticker: str = PRIMARY_TICKER,
) -> dict[str, Any]:
    """Navigation metadata for rewind, calendar, and per-day slice pickers."""
    if not timestamps:
        return {
            "available_dates": [],
            "day_slices": [],
            "selected_date": None,
            "selected_ts": None,
            "prev_ts": None,
            "next_ts": None,
            "is_latest": True,
            "slice_position": 0,
            "slice_count": 0,
        }

    if selected_ts:
        active_ts = selected_ts
    else:
        active_ts = timestamps[-1]
    replay_index = timestamps.index(active_ts) if active_ts in timestamps else max(0, len(timestamps) - 1)
    selected_date = ts_market_date(active_ts)

    return {
        "available_dates": available_dates(timestamps, ticker=ticker),
        "day_slices": build_slice_options(timestamps, selected_date, ticker=ticker),
        "selected_date": selected_date,
        "selected_ts": active_ts,
        "prev_ts": timestamps[replay_index - 1] if active_ts in timestamps and replay_index > 0 else None,
        "next_ts": timestamps[replay_index + 1] if active_ts in timestamps and replay_index + 1 < len(timestamps) else None,
        "is_latest": active_ts == timestamps[-1] if timestamps else True,
        "slice_position": replay_index + 1,
        "slice_count": len(timestamps),
        "day_slice_index": slices_for_date(timestamps, selected_date, ticker=ticker).index(active_ts) + 1
        if active_ts in slices_for_date(timestamps, selected_date, ticker=ticker)
        else 1,
        "day_slice_count": len(slices_for_date(timestamps, selected_date, ticker=ticker)),
    }


def _strike_window(series: pd.Series, spot: float, window_pct: float = 0.04) -> pd.Series:
    if series.empty or spot <= 0:
        return series.sort_index()
    lo, hi = spot * (1 - window_pct), spot * (1 + window_pct)
    window = series.loc[(series.index >= lo) & (series.index <= hi)]
    return window.sort_index() if len(window) >= 5 else series.sort_index()


def _exposure_series(
    greek_df: pd.DataFrame | None,
    strike_series: pd.Series | None,
    exposure: str,
) -> pd.Series:
    exposure = exposure.lower()
    if greek_df is not None and not greek_df.empty:
        df = greek_df.set_index("strike") if "strike" in greek_df.columns else greek_df
        if exposure == "gamma":
            if "net_gex" in df.columns:
                return pd.to_numeric(df["net_gex"], errors="coerce").dropna()
            if "call_gex" in df.columns and "put_gex" in df.columns:
                return pd.to_numeric(df["call_gex"], errors="coerce").fillna(0.0) + pd.to_numeric(
                    df["put_gex"], errors="coerce"
                ).fillna(0.0)
        elif exposure == "vanna" and "call_vanna" in df.columns:
            return pd.to_numeric(df["call_vanna"], errors="coerce").fillna(0.0) + pd.to_numeric(
                df["put_vanna"], errors="coerce"
            ).fillna(0.0)
        elif exposure == "charm" and "call_charm" in df.columns:
            return pd.to_numeric(df["call_charm"], errors="coerce").fillna(0.0) + pd.to_numeric(
                df["put_charm"], errors="coerce"
            ).fillna(0.0)
    if exposure == "gamma" and strike_series is not None:
        return pd.Series(strike_series, dtype=float)
    return pd.Series(dtype=float)


def _mm_positions(greek_df: pd.DataFrame | None) -> dict[str, float]:
    """Summarize net dealer call/put positioning from UW greek table."""
    out = {
        "net_call_delta_bn": 0.0,
        "net_put_delta_bn": 0.0,
        "net_call_gex_bn": 0.0,
        "net_put_gex_bn": 0.0,
    }
    if greek_df is None or greek_df.empty:
        return out
    df = greek_df
    if "call_delta" in df.columns:
        out["net_call_delta_bn"] = float(pd.to_numeric(df["call_delta"], errors="coerce").fillna(0.0).sum())
    if "put_delta" in df.columns:
        out["net_put_delta_bn"] = float(pd.to_numeric(df["put_delta"], errors="coerce").fillna(0.0).sum())
    if "call_gex" in df.columns:
        out["net_call_gex_bn"] = float(pd.to_numeric(df["call_gex"], errors="coerce").fillna(0.0).sum())
    if "put_gex" in df.columns:
        out["net_put_gex_bn"] = float(pd.to_numeric(df["put_gex"], errors="coerce").fillna(0.0).sum())
    return out


def build_periscope_context(
    *,
    ticker: str = PRIMARY_TICKER,
    selected_ts: str | None = None,
    selected_date: str | None = None,
    exposure: str = "gamma",
    uw_entry: dict | None = None,
    history: list[dict] | None = None,
    price_points: list[dict] | None = None,
    previous_snapshot: dict | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Assemble template/API payload for the Periscope exposure view."""
    exposure = exposure.lower() if exposure.lower() in EXPOSURE_TYPES else "gamma"
    api_key = api_key or uw_api_key()
    timestamps = list_periscope_timestamps(ticker, api_key=api_key)
    resolved_ts = resolve_selected_timestamp(timestamps, ts=selected_ts, date=selected_date)
    timeline = build_timeline_navigation(timestamps, resolved_ts, ticker=ticker)
    active_date = selected_date or (ts_market_date(resolved_ts) if resolved_ts else market_today())

    selected = load_periscope_snapshot(
        ticker,
        resolved_ts,
        api_key=api_key,
        uw_entry=uw_entry,
        market_date=active_date,
    ) or {}

    if not price_points:
        price_points = periscope_price_points(
            ticker,
            market_date=active_date,
            api_key=api_key,
            fallback_history=history,
        )

    if previous_snapshot is None and resolved_ts and resolved_ts in timestamps:
        prev_idx = timestamps.index(resolved_ts) - 1
        if prev_idx >= 0:
            prev_ts = timestamps[prev_idx]
            previous_snapshot = load_periscope_snapshot(
                ticker,
                prev_ts,
                api_key=api_key,
                market_date=ts_market_date(prev_ts),
            )

    spot = safe_float(selected.get("spot"), 0.0)
    if uw_entry and uw_entry.get("spot"):
        spot = safe_float(uw_entry["spot"], spot)

    strike_series = selected.get("strike")
    if isinstance(strike_series, pd.Series):
        gex_series = strike_series
    elif strike_series is not None:
        gex_series = pd.Series(strike_series, dtype=float)
    else:
        gex_series = pd.Series(dtype=float)

    greek_df = None
    if uw_entry and uw_entry.get("agg") is not None:
        greek_df = uw_entry["agg"].gex_by_strike.attrs.get("greek_exposure_df")
        if greek_df is None and uw_entry["agg"].surface_data is not None and not uw_entry["agg"].surface_data.empty:
            greek_df = uw_entry["agg"].surface_data

    current_exposure = _exposure_series(greek_df, gex_series, exposure)
    previous_exposure = pd.Series(dtype=float)
    if previous_snapshot:
        prev_strike = previous_snapshot.get("strike")
        if isinstance(prev_strike, pd.Series):
            previous_exposure = _exposure_series(None, prev_strike, exposure)

    gamma_flip = selected.get("gamma_flip")
    if gamma_flip is None and not gex_series.empty:
        cumulative = selected.get("cumulative")
        if isinstance(cumulative, pd.Series) and not cumulative.empty:
            gamma_flip = estimate_gamma_flip(cumulative)

    regime = selected.get("regime", "N/A")
    total_gex = safe_float(selected.get("total_gex"), 0.0)
    if uw_entry and uw_entry.get("agg") is not None:
        total_gex = safe_float(uw_entry["agg"].total_gex_bn, total_gex)
        regime = "LONG gamma" if total_gex >= 0 else "SHORT gamma"

    replay_index = max(0, timestamps.index(selected["ts"])) if selected.get("ts") in timestamps else max(0, len(timestamps) - 1)

    return {
        "ticker": ticker,
        "exposure": exposure,
        "spot": spot or None,
        "regime": regime,
        "total_gex": total_gex,
        "gamma_flip": gamma_flip,
        "call_wall": selected.get("call_wall"),
        "put_wall": selected.get("put_wall"),
        "selected_ts": selected.get("ts"),
        "selected_date": timeline.get("selected_date"),
        "selected_label": selected.get("ts_label", "Latest"),
        "timestamps": timestamps,
        "replay_index": replay_index,
        "prev_ts": timeline.get("prev_ts"),
        "next_ts": timeline.get("next_ts"),
        "timeline": timeline,
        "has_history": bool(timestamps),
        "price_points": price_points or [],
        "exposure_series": current_exposure,
        "previous_exposure": previous_exposure,
        "exposure_window": _strike_window(current_exposure, spot or 0.0, window_pct=0.035),
        "exposure_extended": _strike_window(current_exposure, spot or 0.0, window_pct=0.08),
        "mm_positions": _mm_positions(greek_df if isinstance(greek_df, pd.DataFrame) else None),
        "history": history or [],
        "selected": selected,
        "data_path": "uw_api" if should_use_api_for_date(active_date, api_key=api_key) else "exports",
        "uw_fetched_at": uw_entry.get("fetched_at") if uw_entry else None,
        "vanna_charm_available": bool(
            isinstance(greek_df, pd.DataFrame)
            and {"call_vanna", "put_vanna", "call_charm", "put_charm"}.intersection(greek_df.columns)
        ),
    }
