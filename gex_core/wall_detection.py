"""Gamma wall detection that ignores dust strikes."""

from __future__ import annotations

import os

import pandas as pd

from gex_core.features import safe_float, select_atm_strike_series


def wall_min_gex_share() -> float:
    try:
        return float(os.environ.get("GEX_WALL_MIN_GEX_SHARE", "0.01"))
    except (TypeError, ValueError):
        return 0.01


def detect_walls(
    gex_by_strike: pd.Series,
    spot: float,
    *,
    window_pct: float = 0.12,
) -> dict[str, float | None]:
    """Return call/put walls from meaningful |GEX| peaks near spot."""
    strike = pd.Series(gex_by_strike, dtype=float).sort_index()
    if strike.empty or spot <= 0:
        return {"call_wall": None, "put_wall": None, "pos_gamma_peak_strike": None}

    near = select_atm_strike_series(strike, spot, window_pct=window_pct, min_strikes=3)
    if near.empty:
        near = strike

    total_abs = float(near.abs().sum())
    min_abs = max(total_abs * wall_min_gex_share(), 1e-8)
    meaningful = near[near.abs() >= min_abs]
    if meaningful.empty:
        meaningful = near

    positive = meaningful[meaningful > 0]
    negative = meaningful[meaningful < 0]
    call_wall = float(positive.idxmax()) if len(positive) else None
    put_wall = float(negative.idxmin()) if len(negative) else None
    pos_peak = call_wall
    return {
        "call_wall": call_wall,
        "put_wall": put_wall,
        "pos_gamma_peak_strike": pos_peak,
    }
