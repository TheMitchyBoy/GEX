"""Periscope-style market maker exposure dashboard data assembly."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pandas as pd

from gex_core.env_bootstrap import uw_api_key
from gex_core.features import (
    greek_gamma_series_from_df,
    magnet_gamma_from_call_put,
    safe_float,
    select_atm_strike_series,
    spot_covers_strike_grid,
)
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
from gex_core.spot_exposure import spot_exposure_mm_positions, spot_exposure_net_series
from gex_core.tickers import PRIMARY_TICKER

logger = logging.getLogger(__name__)

EXPOSURE_TYPES = ("gamma", "vanna", "charm")
PERISCOPE_PROFILE_MAX = 55
PERISCOPE_EXTENDED_MAX = 96

_GREEK_FETCH_CACHE: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
_GREEK_FETCH_LOCK = threading.Lock()
_GREEK_FETCH_TTL = 120.0


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


def _strike_window(
    series: pd.Series,
    spot: float,
    window_pct: float = 0.04,
    *,
    max_strikes: int | None = None,
) -> pd.Series:
    return select_atm_strike_series(
        series,
        spot,
        window_pct=window_pct,
        min_strikes=5,
        max_strikes=max_strikes,
    )


def _greek_exposure_from_df(greek_df: pd.DataFrame | None, exposure: str) -> pd.Series:
    if greek_df is None or greek_df.empty:
        return pd.Series(dtype=float)
    df = greek_df.set_index("strike") if "strike" in greek_df.columns else greek_df
    if exposure == "gamma":
        return greek_gamma_series_from_df(greek_df)
    elif exposure == "vanna" and "call_vanna" in df.columns:
        return pd.to_numeric(df["call_vanna"], errors="coerce").fillna(0.0) + pd.to_numeric(
            df["put_vanna"], errors="coerce"
        ).fillna(0.0)
    elif exposure == "charm" and "call_charm" in df.columns:
        return pd.to_numeric(df["call_charm"], errors="coerce").fillna(0.0) + pd.to_numeric(
            df["put_charm"], errors="coerce"
        ).fillna(0.0)
    return pd.Series(dtype=float)


def _snapshot_strike_is_spot_oi(snapshot: dict[str, Any]) -> bool:
    """True when snapshot strike came from UW spot-exposures (OI), not greek-exposure."""
    spot_df = snapshot.get("spot_exposures_df")
    if isinstance(spot_df, pd.DataFrame) and not spot_df.empty:
        return True
    endpoint = str(snapshot.get("uw_endpoint") or "").lower()
    return "spot-exposure" in endpoint


def _greek_df_has_call_put(greek_df: pd.DataFrame | None) -> bool:
    if greek_df is None or greek_df.empty:
        return False
    return {"call_gex", "put_gex"}.issubset(greek_df.columns)


def _greek_df_from_surface(surface: pd.DataFrame | None) -> pd.DataFrame | None:
    if surface is None or surface.empty or "strike" not in surface.columns:
        return None
    if _greek_df_has_call_put(surface):
        out = surface.copy()
        if "GEX" in out.columns and "net_gex" not in out.columns:
            out["net_gex"] = pd.to_numeric(out["GEX"], errors="coerce")
        return out
    if "net_gex" in surface.columns or "GEX" in surface.columns:
        out = surface.copy()
        if "GEX" in out.columns and "net_gex" not in out.columns:
            out["net_gex"] = pd.to_numeric(out["GEX"], errors="coerce")
        return out
    return None


def _fetch_greek_exposure_cached(
    ticker: str,
    market_date: str,
    *,
    api_key: str | None,
) -> pd.DataFrame | None:
    """Best-effort greek-exposure/strike fetch for magnet maps (short TTL cache)."""
    if not api_key or not market_date:
        return None
    ticker = ticker.upper()
    market_date = market_date[:10]
    cache_key = (ticker, market_date)
    with _GREEK_FETCH_LOCK:
        cached = _GREEK_FETCH_CACHE.get(cache_key)
        if cached and (time.monotonic() - cached[0]) < _GREEK_FETCH_TTL:
            return cached[1].copy()

    try:
        from gex_core.uw_loader import fetch_uw_greek_exposure

        df = fetch_uw_greek_exposure(ticker, api_key=api_key, date=market_date)
    except Exception:
        logger.debug("Magnet greek-exposure fetch failed for %s on %s", ticker, market_date, exc_info=True)
        return None

    if df is None or df.empty:
        return None
    with _GREEK_FETCH_LOCK:
        _GREEK_FETCH_CACHE[cache_key] = (time.monotonic(), df.copy())
    return df


def _ensure_greek_exposure_df(
    *,
    ticker: str,
    market_date: str | None,
    api_key: str | None,
    snapshot: dict[str, Any],
    uw_entry: dict | None,
    greek_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Resolve call/put greek data for magnet maps — never spot-exposures OI."""
    if _greek_df_has_call_put(greek_df):
        return greek_df

    snap_df = snapshot.get("greek_exposure_df")
    if isinstance(snap_df, pd.DataFrame) and not snap_df.empty:
        if _greek_df_has_call_put(snap_df):
            return snap_df
        if greek_df is None:
            greek_df = snap_df

    if uw_entry and uw_entry.get("agg") is not None:
        attrs_df = uw_entry["agg"].gex_by_strike.attrs.get("greek_exposure_df")
        if isinstance(attrs_df, pd.DataFrame) and not attrs_df.empty:
            if _greek_df_has_call_put(attrs_df):
                return attrs_df
            if greek_df is None:
                greek_df = attrs_df
        surface_df = _greek_df_from_surface(uw_entry["agg"].surface_data)
        if surface_df is not None and _greek_df_has_call_put(surface_df):
            return surface_df

    surface_df = _greek_df_from_surface(snapshot.get("surface_df"))
    if surface_df is not None and _greek_df_has_call_put(surface_df):
        return surface_df

    greek_path = snapshot.get("greek_exposure_path")
    if greek_path is not None:
        from gex_core.exports import load_greek_exposure_df

        loaded = load_greek_exposure_df(greek_path)
        if _greek_df_has_call_put(loaded):
            return loaded
        if greek_df is None and not loaded.empty:
            greek_df = loaded

    if market_date:
        fetched = _fetch_greek_exposure_cached(ticker, market_date, api_key=api_key)
        if fetched is not None and not fetched.empty:
            return fetched

    return greek_df if isinstance(greek_df, pd.DataFrame) and not greek_df.empty else None


def _magnet_net_fallback_series(
    *,
    exposure: str,
    snapshot: dict[str, Any],
    uw_entry: dict | None,
    greek_df: pd.DataFrame | None,
    gex_series: pd.Series,
) -> pd.Series:
    """Net greek profile for magnets when call/put decomposition is unavailable."""
    exposure = exposure.lower()
    greek_strike = snapshot.get("greek_strike")
    if isinstance(greek_strike, pd.Series) and not greek_strike.empty:
        return greek_strike.sort_index()

    if uw_entry and uw_entry.get("agg") is not None:
        gbs = uw_entry["agg"].gex_by_strike
        if isinstance(gbs, pd.Series) and not gbs.empty:
            return pd.Series(gbs, dtype=float).sort_index()

    if isinstance(greek_df, pd.DataFrame) and not greek_df.empty:
        series = _greek_exposure_from_df(greek_df, exposure)
        if not series.empty:
            return series.sort_index()

    if exposure == "gamma" and not gex_series.empty:
        return gex_series.sort_index()

    return pd.Series(dtype=float)


def _resolve_magnet_greek_df(
    *,
    uw_entry: dict | None,
    greek_df: pd.DataFrame | None,
    snapshot: dict[str, Any],
) -> pd.DataFrame | None:
    if isinstance(greek_df, pd.DataFrame) and not greek_df.empty:
        return greek_df
    snap_df = snapshot.get("greek_exposure_df")
    if isinstance(snap_df, pd.DataFrame) and not snap_df.empty:
        return snap_df
    if uw_entry and uw_entry.get("agg") is not None:
        attrs_df = uw_entry["agg"].gex_by_strike.attrs.get("greek_exposure_df")
        if isinstance(attrs_df, pd.DataFrame) and not attrs_df.empty:
            return attrs_df
        surface = _greek_df_from_surface(uw_entry["agg"].surface_data)
        if surface is not None:
            return surface
    surface_df = _greek_df_from_surface(snapshot.get("surface_df"))
    if surface_df is not None:
        return surface_df
    snap_greek = snapshot.get("greek_exposure_df")
    if isinstance(snap_greek, pd.DataFrame) and not snap_greek.empty:
        return snap_greek
    return None


def _resolve_spot_exposure_df(
    *,
    uw_entry: dict | None,
    snapshot: dict[str, Any],
) -> pd.DataFrame | None:
    if uw_entry and uw_entry.get("agg") is not None:
        spot_df = uw_entry["agg"].gex_by_strike.attrs.get("spot_exposures_df")
        if isinstance(spot_df, pd.DataFrame) and not spot_df.empty:
            return spot_df
    spot_df = snapshot.get("spot_exposures_df")
    if isinstance(spot_df, pd.DataFrame) and not spot_df.empty:
        return spot_df
    return None


def _magnet_exposure_series(
    *,
    exposure: str,
    spot: float | None,
    uw_entry: dict | None,
    greek_df: pd.DataFrame | None,
    snapshot: dict[str, Any],
    gex_series: pd.Series,
) -> pd.Series:
    """Magnet map profile — primary source is spot-exposures/strike OI."""
    from gex_core.spot_exposure import spot_exposure_surface_df

    exposure = exposure.lower()
    spot_df = _resolve_spot_exposure_df(uw_entry=uw_entry, snapshot=snapshot)
    if isinstance(spot_df, pd.DataFrame) and not spot_df.empty:
        surface = spot_exposure_surface_df(spot_df, exposure)
        if exposure == "gamma" and not surface.empty:
            series = magnet_gamma_from_call_put(surface, spot)
            if not series.empty:
                return series.sort_index()
        if not surface.empty and "net_gex" in surface.columns:
            return pd.Series(
                pd.to_numeric(surface.set_index("strike")["net_gex"], errors="coerce"),
                dtype=float,
            ).dropna().sort_index()

    resolved_df = _resolve_magnet_greek_df(uw_entry=uw_entry, greek_df=greek_df, snapshot=snapshot)
    if exposure == "gamma" and isinstance(resolved_df, pd.DataFrame) and not resolved_df.empty:
        series = magnet_gamma_from_call_put(resolved_df, spot)
        if not series.empty:
            return series.sort_index()

    return _magnet_net_fallback_series(
        exposure=exposure,
        snapshot=snapshot,
        uw_entry=uw_entry,
        greek_df=resolved_df,
        gex_series=gex_series,
    )


def _exposure_series(
    spot_df: pd.DataFrame | None,
    greek_df: pd.DataFrame | None,
    strike_series: pd.Series | None,
    exposure: str,
    *,
    spot: float | None = None,
) -> pd.Series:
    """Per-strike exposure — primary source is spot-exposures/strike."""
    exposure = exposure.lower()
    spot_val = safe_float(spot, 0.0)
    spot_series = spot_exposure_net_series(spot_df, exposure)
    greek_series = _greek_exposure_from_df(greek_df, exposure)

    if spot_val > 0 and not spot_covers_strike_grid(spot_series, spot_val) and not greek_series.empty:
        return greek_series.sort_index()
    if not spot_series.empty:
        return spot_series.sort_index()
    if not greek_series.empty:
        return greek_series.sort_index()
    if exposure == "gamma" and strike_series is not None:
        return pd.Series(strike_series, dtype=float).sort_index()
    return pd.Series(dtype=float)


def _mm_positions(spot_df: pd.DataFrame | None, greek_df: pd.DataFrame | None) -> dict[str, float]:
    """Summarize net dealer call/put positioning from UW spot-exposures/strike."""
    spot_positions = spot_exposure_mm_positions(spot_df)
    if any(abs(v) > 1e-12 for v in spot_positions.values()):
        return spot_positions

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

    is_latest_slice = resolved_ts == timestamps[-1] if timestamps and resolved_ts else True
    spot = safe_float(selected.get("spot"), 0.0)
    if uw_entry and uw_entry.get("spot") and is_latest_slice:
        spot = safe_float(uw_entry["spot"], spot)

    strike_series = selected.get("strike")
    if isinstance(strike_series, pd.Series):
        gex_series = strike_series
    elif strike_series is not None:
        gex_series = pd.Series(strike_series, dtype=float)
    else:
        gex_series = pd.Series(dtype=float)

    spot_df = selected.get("spot_exposures_df")
    greek_df = selected.get("greek_exposure_df")
    if not isinstance(greek_df, pd.DataFrame) or greek_df.empty:
        greek_df = None
    if uw_entry and uw_entry.get("agg") is not None:
        if not isinstance(spot_df, pd.DataFrame) or spot_df.empty:
            spot_df = uw_entry["agg"].gex_by_strike.attrs.get("spot_exposures_df")
        if greek_df is None:
            greek_df = uw_entry["agg"].gex_by_strike.attrs.get("greek_exposure_df")
        if greek_df is None:
            greek_df = _greek_df_from_surface(uw_entry["agg"].surface_data)

    greek_df = _ensure_greek_exposure_df(
        ticker=ticker,
        market_date=active_date,
        api_key=api_key,
        snapshot=selected,
        uw_entry=uw_entry,
        greek_df=greek_df,
    )

    current_exposure = _exposure_series(spot_df, greek_df, gex_series, exposure, spot=spot or None)
    previous_exposure = pd.Series(dtype=float)
    if previous_snapshot:
        prev_spot = previous_snapshot.get("spot_exposures_df")
        prev_strike = previous_snapshot.get("strike")
        prev_spot_price = safe_float(previous_snapshot.get("spot"), spot)
        if isinstance(prev_spot, pd.DataFrame) and not prev_spot.empty:
            previous_exposure = _exposure_series(prev_spot, None, None, exposure, spot=prev_spot_price or None)
        elif isinstance(prev_strike, pd.Series):
            previous_exposure = _exposure_series(None, None, prev_strike, exposure, spot=prev_spot_price or None)

    # UW Periscope profile: native spot-exposures grid (~50 ATM strikes).
    exposure_profile = current_exposure.sort_index()
    if len(exposure_profile) > PERISCOPE_PROFILE_MAX:
        exposure_profile = _strike_window(
            exposure_profile, spot or 0.0, window_pct=0.045, max_strikes=PERISCOPE_PROFILE_MAX
        )

    magnet_exposure = current_exposure
    previous_magnet_exposure = previous_exposure
    exposure_extended = _strike_window(
        current_exposure,
        spot or 0.0,
        window_pct=0.085,
        max_strikes=PERISCOPE_EXTENDED_MAX,
    )
    gamma_flip = None

    total_gex = safe_float(selected.get("total_gex"), 0.0)
    if total_gex == 0 and not current_exposure.empty:
        total_gex = float(current_exposure.sum())
    elif total_gex == 0 and not gex_series.empty:
        total_gex = float(gex_series.sum())
    regime = selected.get("regime", "N/A")
    if total_gex != 0:
        regime = "LONG gamma" if total_gex >= 0 else "SHORT gamma"

    replay_index = max(0, timestamps.index(selected["ts"])) if selected.get("ts") in timestamps else max(0, len(timestamps) - 1)

    exposure_trail: list[dict[str, Any]] = []
    if resolved_ts and resolved_ts in timestamps:
        trail_idx = timestamps.index(resolved_ts)
        for back in range(1, 5):
            prior_idx = trail_idx - back
            if prior_idx < 0:
                break
            trail_ts = timestamps[prior_idx]
            trail_snap = load_periscope_snapshot(
                ticker,
                trail_ts,
                api_key=api_key,
                market_date=ts_market_date(trail_ts),
            )
            if not trail_snap:
                continue
            trail_spot = safe_float(trail_snap.get("spot"), 0.0)
            trail_strike = trail_snap.get("strike")
            trail_spot_df = trail_snap.get("spot_exposures_df")
            trail_greek_df = trail_snap.get("greek_exposure_df")
            trail_series = _exposure_series(
                trail_spot_df if isinstance(trail_spot_df, pd.DataFrame) else None,
                trail_greek_df if isinstance(trail_greek_df, pd.DataFrame) else None,
                trail_strike if isinstance(trail_strike, pd.Series) else None,
                exposure,
                spot=trail_spot or None,
            )
            if trail_series.empty:
                continue
            exposure_trail.append(
                {
                    "ts": trail_ts,
                    "label": trail_snap.get("ts_label") or ts_market_time_label(trail_ts),
                    "spot": trail_spot,
                    "series": trail_series.sort_index(),
                    "age": back,
                }
            )
        exposure_trail.reverse()

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
        "magnet_exposure_series": magnet_exposure.sort_index(),
        "previous_exposure": previous_exposure,
        "exposure_trail": exposure_trail,
        "previous_magnet_exposure": previous_magnet_exposure.sort_index()
        if isinstance(previous_magnet_exposure, pd.Series) and not previous_magnet_exposure.empty
        else previous_exposure,
        "exposure_profile": exposure_profile,
        "exposure_window": exposure_profile,
        "exposure_extended": exposure_extended,
        "mm_positions": _mm_positions(
            spot_df if isinstance(spot_df, pd.DataFrame) else None,
            greek_df if isinstance(greek_df, pd.DataFrame) else None,
        ),
        "history": history or [],
        "selected": selected,
        "data_path": "uw_api" if should_use_api_for_date(active_date, api_key=api_key) else "exports",
        "uw_fetched_at": uw_entry.get("fetched_at") if uw_entry else None,
        "vanna_charm_available": bool(
            (isinstance(spot_df, pd.DataFrame) and {"call_vanna_oi", "put_vanna_oi"}.issubset(spot_df.columns))
            or (
                isinstance(greek_df, pd.DataFrame)
                and {"call_vanna", "put_vanna", "call_charm", "put_charm"}.intersection(greek_df.columns)
            )
        ),
    }
