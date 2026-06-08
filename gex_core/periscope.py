"""Periscope-style market maker exposure dashboard data assembly."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gex_core.features import safe_float
from gex_core.env_bootstrap import uw_api_key
from gex_core.features import gamma_flip_from_profile, select_atm_strike_series, spot_covers_strike_grid
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

EXPOSURE_TYPES = ("gamma", "vanna", "charm")
PERISCOPE_PROFILE_MAX = 55
PERISCOPE_EXTENDED_MAX = 96


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
        if "net_gex" in df.columns:
            return pd.to_numeric(df["net_gex"], errors="coerce").dropna()
        if "GEX" in df.columns:
            return pd.to_numeric(df["GEX"], errors="coerce").dropna()
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
    return pd.Series(dtype=float)


def _magnet_gamma_from_call_put(greek_df: pd.DataFrame | None, spot: float | None) -> pd.Series:
    """Magnet profile: when call/put gamma cancel, show the dominant leg near spot."""
    if greek_df is None or greek_df.empty:
        return pd.Series(dtype=float)
    df = greek_df.set_index("strike") if "strike" in greek_df.columns else greek_df
    if "call_gex" not in df.columns or "put_gex" not in df.columns:
        return _greek_exposure_from_df(greek_df, "gamma")

    calls = pd.to_numeric(df["call_gex"], errors="coerce").fillna(0.0)
    puts = pd.to_numeric(df["put_gex"], errors="coerce").fillna(0.0)
    if "net_gex" in df.columns:
        net = pd.to_numeric(df["net_gex"], errors="coerce").fillna(calls + puts)
    elif "GEX" in df.columns:
        net = pd.to_numeric(df["GEX"], errors="coerce").fillna(calls + puts)
    else:
        net = calls + puts

    spot_val = safe_float(spot, 0.0)
    values: dict[float, float] = {}
    for strike in df.index:
        strike_f = float(strike)
        n = float(net.loc[strike])
        c = float(calls.loc[strike])
        p = float(puts.loc[strike])
        leg_peak = max(abs(c), abs(p))
        if leg_peak >= 0.1 and abs(n) < 0.15 * leg_peak:
            if spot_val > 0 and strike_f < spot_val and c > abs(p):
                values[strike_f] = c
            elif spot_val > 0 and strike_f > spot_val and abs(p) > c:
                values[strike_f] = p
            else:
                values[strike_f] = n
        else:
            values[strike_f] = n
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values, dtype=float).sort_index()


def _snapshot_strike_is_spot_oi(snapshot: dict[str, Any]) -> bool:
    """True when snapshot strike came from UW spot-exposures (OI), not greek-exposure."""
    spot_df = snapshot.get("spot_exposures_df")
    return isinstance(spot_df, pd.DataFrame) and not spot_df.empty


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
        surface = uw_entry["agg"].surface_data
        if isinstance(surface, pd.DataFrame) and not surface.empty:
            return surface
    surface_df = snapshot.get("surface_df")
    if isinstance(surface_df, pd.DataFrame) and not surface_df.empty:
        return surface_df
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
    """Greek-exposure profile for the magnet map — never spot-exposures OI when greek exists."""
    exposure = exposure.lower()
    resolved_df = _resolve_magnet_greek_df(uw_entry=uw_entry, greek_df=greek_df, snapshot=snapshot)

    if exposure == "gamma" and isinstance(resolved_df, pd.DataFrame) and not resolved_df.empty:
        series = _magnet_gamma_from_call_put(resolved_df, spot)
        if not series.empty:
            return series.sort_index()

    series = _greek_exposure_from_df(resolved_df, exposure)
    if not series.empty:
        return series.sort_index()

    if uw_entry and uw_entry.get("agg") is not None:
        gbs = uw_entry["agg"].gex_by_strike
        if isinstance(gbs, pd.Series) and not gbs.empty:
            return pd.Series(gbs, dtype=float).sort_index()

    greek_strike = snapshot.get("greek_strike")
    if isinstance(greek_strike, pd.Series) and not greek_strike.empty:
        return greek_strike.sort_index()
    if greek_strike is not None and not isinstance(greek_strike, pd.Series):
        converted = pd.Series(greek_strike, dtype=float)
        if not converted.empty:
            return converted.sort_index()

    if exposure == "gamma" and not gex_series.empty and not _snapshot_strike_is_spot_oi(snapshot):
        return gex_series.sort_index()

    return pd.Series(dtype=float)


def _prefer_denser_exposure(spot_series: pd.Series, greek_series: pd.Series, spot: float) -> pd.Series | None:
    """Prefer the greek chain when it has materially more strikes near spot."""
    spot_val = safe_float(spot, 0.0)
    if spot_val <= 0 or spot_series.empty or greek_series.empty:
        return None
    band = spot_val * 0.025
    lo, hi = spot_val - band, spot_val + band
    spot_n = int(((spot_series.index >= lo) & (spot_series.index <= hi)).sum())
    greek_n = int(((greek_series.index >= lo) & (greek_series.index <= hi)).sum())
    if greek_n >= max(spot_n + 8, int(spot_n * 1.35)):
        return greek_series.sort_index()
    return None


def _exposure_series(
    spot_df: pd.DataFrame | None,
    greek_df: pd.DataFrame | None,
    strike_series: pd.Series | None,
    exposure: str,
    *,
    spot: float | None = None,
) -> pd.Series:
    """Per-strike exposure — prefer UW spot-exposures when they bracket spot."""
    exposure = exposure.lower()
    spot_val = safe_float(spot, 0.0)
    spot_series = spot_exposure_net_series(spot_df, exposure)
    greek_series = _greek_exposure_from_df(greek_df, exposure)

    if spot_val > 0 and not spot_covers_strike_grid(spot_series, spot_val) and not greek_series.empty:
        return greek_series.sort_index()
    denser = _prefer_denser_exposure(spot_series, greek_series, spot_val)
    if denser is not None:
        return denser
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
        if greek_df is None and uw_entry["agg"].surface_data is not None and not uw_entry["agg"].surface_data.empty:
            greek_df = uw_entry["agg"].surface_data

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

    # Extended panel: wider greek-exposure chain when available.
    greek_exposure = _magnet_exposure_series(
        exposure=exposure,
        spot=spot or None,
        uw_entry=uw_entry,
        greek_df=greek_df,
        snapshot=selected,
        gex_series=gex_series,
    )
    magnet_exposure = greek_exposure if not greek_exposure.empty else current_exposure
    previous_magnet_exposure = previous_exposure
    if previous_snapshot:
        prev_greek = _magnet_exposure_series(
            exposure=exposure,
            spot=safe_float(previous_snapshot.get("spot"), spot) or None,
            uw_entry=None,
            greek_df=None,
            snapshot=previous_snapshot,
            gex_series=previous_snapshot.get("strike")
            if isinstance(previous_snapshot.get("strike"), pd.Series)
            else pd.Series(dtype=float),
        )
        if not prev_greek.empty:
            previous_magnet_exposure = prev_greek
    extended_source = greek_exposure if not greek_exposure.empty else current_exposure
    exposure_extended = _strike_window(
        extended_source,
        spot or 0.0,
        window_pct=0.085,
        max_strikes=PERISCOPE_EXTENDED_MAX,
    )

    gamma_flip = None
    if not magnet_exposure.empty and spot > 0:
        gamma_flip = gamma_flip_from_profile(magnet_exposure, spot)
    elif not current_exposure.empty and spot > 0:
        gamma_flip = gamma_flip_from_profile(current_exposure, spot)
    if gamma_flip is None:
        gamma_flip = selected.get("gamma_flip")
    if gamma_flip is None and not gex_series.empty:
        gamma_flip = gamma_flip_from_profile(gex_series, spot or None)

    regime = selected.get("regime", "N/A")
    total_gex = safe_float(selected.get("total_gex"), 0.0)
    if uw_entry and uw_entry.get("spot_gamma_bn") is not None:
        if is_latest_slice:
            total_gex = safe_float(uw_entry["spot_gamma_bn"], total_gex)
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
        "magnet_exposure_series": magnet_exposure.sort_index(),
        "previous_exposure": previous_exposure,
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
