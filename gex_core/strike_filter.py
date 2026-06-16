"""Filter UW strike profiles to tradeable ranges around spot."""

from __future__ import annotations

import os

import pandas as pd

from gex_core.features import safe_float, select_atm_strike_series


def store_strike_distance_pct() -> float:
    try:
        return float(os.environ.get("GEX_STORE_STRIKE_DISTANCE_PCT", os.environ.get("GEX_MAX_STRIKE_DISTANCE_PCT", "0.12")))
    except (TypeError, ValueError):
        return 0.12


def min_strike_gex_bn() -> float:
    """Drop strikes whose |GEX| is below this (Bn$ / 1% move)."""
    try:
        return float(os.environ.get("GEX_MIN_STRIKE_GEX_BN", "1e-6"))
    except (TypeError, ValueError):
        return 1e-8


def strikes_bracket_spot(
    gex_by_strike: pd.Series,
    spot: float,
    *,
    window_pct: float = 0.05,
    min_strikes: int = 2,
) -> bool:
    """True when enough strikes exist within ``window_pct`` of spot."""
    if gex_by_strike is None or gex_by_strike.empty or spot <= 0:
        return False
    strikes = pd.to_numeric(gex_by_strike.index, errors="coerce")
    lo, hi = spot * (1 - window_pct), spot * (1 + window_pct)
    in_window = int(((strikes >= lo) & (strikes <= hi)).sum())
    return in_window >= min_strikes


def filter_strikes_for_storage(
    gex_by_strike: pd.Series,
    spot: float,
    *,
    window_pct: float | None = None,
    min_abs_gex_bn: float | None = None,
    min_strikes: int = 3,
) -> pd.Series:
    """Keep strikes near spot for ``snapshot_strikes`` (drop far OTM noise)."""
    if gex_by_strike is None or gex_by_strike.empty:
        return pd.Series(dtype=float)
    window_pct = store_strike_distance_pct() if window_pct is None else window_pct
    min_abs = min_strike_gex_bn() if min_abs_gex_bn is None else min_abs_gex_bn
    spot_val = safe_float(spot, 0.0)

    series = pd.Series(gex_by_strike, dtype=float).sort_index()
    if min_abs > 0:
        series = series[series.abs() >= min_abs]
    if series.empty:
        return series

    if spot_val > 0:
        lo, hi = spot_val * (1 - window_pct), spot_val * (1 + window_pct)
        in_window = series.loc[(series.index >= lo) & (series.index <= hi)]
        if len(in_window) >= min_strikes:
            series = in_window
        else:
            series = select_atm_strike_series(
                series,
                spot_val,
                window_pct=window_pct,
                min_strikes=min_strikes,
            )
    return series.sort_index()


def resolve_storage_strike_profile(
    gex_by_strike: pd.Series,
    *,
    spot: float,
    greek_df: pd.DataFrame | None = None,
) -> tuple[pd.Series, str]:
    """Pick a spot-centered strike profile for Postgres storage.

    Falls back to filtered greek-exposure when spot-exposures rows do not
    bracket the current spot (e.g. stale low-strike pages from UW).
    """
    from gex_core.spot_exposure import spot_exposure_net_series

    profile = pd.Series(gex_by_strike, dtype=float).sort_index()
    source = "spot_exposures"

    if spot > 0 and not strikes_bracket_spot(profile, spot):
        if greek_df is not None and not greek_df.empty:
            greek_profile = spot_exposure_net_series(greek_df, "gamma")
            if greek_profile.empty and "net_gex" in greek_df.columns:
                greek_profile = pd.Series(
                    pd.to_numeric(greek_df["net_gex"], errors="coerce").values,
                    index=pd.to_numeric(greek_df["strike"], errors="coerce"),
                    dtype=float,
                ).dropna().sort_index()
            if strikes_bracket_spot(greek_profile, spot):
                profile = greek_profile
                source = "greek_exposure_atm"
            else:
                profile = greek_profile
                source = "greek_exposure_filtered"
        else:
            source = "spot_exposures_misaligned"

    filtered = filter_strikes_for_storage(profile, spot)
    if filtered.empty and not profile.empty and spot > 0:
        filtered = select_atm_strike_series(profile, spot, window_pct=0.15, min_strikes=5)
    if not filtered.empty:
        return filtered.sort_index(), source
    return profile, source
