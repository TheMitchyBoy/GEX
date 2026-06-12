"""Feature engineering for GEX prediction and similarity matching.

Each historical snapshot is reduced to a numeric vector (total GEX, flip
distance, strike concentration, term-structure shape, optional market-context
fields from ``market_features``). ``predict_next_snapshot`` compares the
current vector to past snapshots with exponential recency weighting and
strike-surface cosine similarity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gex_core.exports import (
    load_cumulative_series,
    load_expiration_series,
    load_strike_series,
    load_surface_df,
)
from gex_core.extended_features import EXTENDED_FEATURE_DEFAULTS, apply_extended_defaults, extended_feature_names
from gex_core.event_calendar import event_calendar_features


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_gamma_flip_value(value: Any) -> float | None:
    """Normalize gamma flip from summary JSON (float or legacy detail dict)."""
    if value is None:
        return None
    if isinstance(value, dict):
        flip = safe_float(value.get("flip_strike"), 0.0)
        return flip if flip > 0 else None
    try:
        flip = float(value)
    except (TypeError, ValueError):
        return None
    return flip if flip > 0 else None


def resolve_gamma_flip(
    *,
    spot: float | None = None,
    gex_by_strike: pd.Series | None = None,
    cumulative_gex: pd.Series | None = None,
    greek_exposure_df: pd.DataFrame | None = None,
    greek_strike: pd.Series | None = None,
    spot_exposure_df: pd.DataFrame | None = None,
) -> float | None:
    """Canonical gamma flip: spot-exposures/strike magnet profile near ATM, then fallbacks."""
    from gex_core.spot_exposure import spot_exposure_surface_df

    if spot_exposure_df is not None and not spot_exposure_df.empty:
        surface = spot_exposure_surface_df(spot_exposure_df, "gamma")
        if not surface.empty:
            flip = gamma_flip_from_uw_greek(surface, spot, gex_by_strike=gex_by_strike)
            if flip is not None:
                return flip
    if gex_by_strike is not None and not gex_by_strike.empty:
        flip = gamma_flip_from_profile(gex_by_strike, spot)
        if flip is not None:
            return flip
    if greek_exposure_df is not None and not greek_exposure_df.empty:
        flip = gamma_flip_from_uw_greek(greek_exposure_df, spot, gex_by_strike=gex_by_strike)
        if flip is not None:
            return flip
    if greek_strike is not None and not greek_strike.empty:
        flip = gamma_flip_from_profile(greek_strike, spot)
        if flip is not None:
            return flip
    if cumulative_gex is not None and not cumulative_gex.empty:
        return estimate_gamma_flip(cumulative_gex)
    return None


def estimate_gamma_flip_detailed(
    *,
    spot: float | None = None,
    gex_by_strike: pd.Series | None = None,
    cumulative_gex: pd.Series | None = None,
    greek_exposure_df: pd.DataFrame | None = None,
    greek_strike: pd.Series | None = None,
    spot_exposure_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Gamma flip strike with confidence from the ATM-local profile."""
    from gex_core.spot_exposure import spot_exposure_surface_df

    flip = resolve_gamma_flip(
        spot=spot,
        gex_by_strike=gex_by_strike,
        cumulative_gex=cumulative_gex,
        greek_exposure_df=greek_exposure_df,
        greek_strike=greek_strike,
        spot_exposure_df=spot_exposure_df,
    )
    if flip is None:
        return {"flip_strike": None, "confidence": "none", "message": "no zero-crossing"}

    profile = pd.Series(dtype=float)
    if spot_exposure_df is not None and not spot_exposure_df.empty:
        surface = spot_exposure_surface_df(spot_exposure_df, "gamma")
        if not surface.empty:
            profile = magnet_gamma_from_call_put(surface, spot)
    if profile.empty and gex_by_strike is not None and not gex_by_strike.empty:
        profile = pd.Series(gex_by_strike, dtype=float).sort_index()
    if profile.empty and greek_exposure_df is not None and not greek_exposure_df.empty:
        profile = magnet_gamma_from_call_put(greek_exposure_df, spot)
    if profile.empty and greek_strike is not None and not greek_strike.empty:
        profile = pd.Series(greek_strike, dtype=float).sort_index()
    if profile.empty and cumulative_gex is not None and not cumulative_gex.empty:
        profile = cumulative_gex.diff().dropna()
        if profile.empty:
            profile = cumulative_gex

    local = profile
    spot_val = safe_float(spot, 0.0)
    if spot_val > 0 and len(profile) >= 2:
        windowed = select_atm_strike_series(profile, spot_val, window_pct=0.06, min_strikes=5)
        if len(windowed) >= 2:
            local = windowed

    cumulative = local.cumsum()
    signs = np.sign(cumulative.astype(float).values)
    change_points = np.where(np.diff(signs) != 0)[0]
    local_slope = 0.0
    if len(change_points):
        idx = int(change_points[0])
        x0 = float(cumulative.index[idx])
        x1 = float(cumulative.index[idx + 1])
        y0 = float(cumulative.iloc[idx])
        y1 = float(cumulative.iloc[idx + 1])
        local_slope = abs(y1 - y0) / max(abs(x1 - x0), 1e-9)

    if local_slope >= 0.10:
        confidence = "high"
    elif local_slope >= 0.03:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "flip_strike": float(flip),
        "confidence": confidence,
        "message": "ok",
        "local_slope": float(local_slope),
    }


def estimate_gamma_flip(cumulative: pd.Series) -> float | None:
    if cumulative.empty:
        return None
    values = cumulative.astype(float).values
    idx = pd.to_numeric(cumulative.index, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(idx)
    if int(np.sum(valid)) < 2:
        return None
    x = idx[valid]
    y = values[valid]
    signs = np.sign(y)
    for i in range(len(signs) - 1):
        if signs[i] == signs[i + 1]:
            continue
        x0, x1 = float(x[i]), float(x[i + 1])
        y0, y1 = float(y[i]), float(y[i + 1])
        if y1 == y0:
            return x0
        return x0 - y0 * (x1 - x0) / (y1 - y0)
    return None


def gamma_flip_from_profile(
    strike_series: pd.Series | None,
    spot: float | None = None,
    *,
    window_pct: float = 0.06,
    min_strikes: int = 5,
) -> float | None:
    """Gamma flip from an ATM-local cumulative profile.

    The full strike chain often crosses zero only far OTM; dealers flip near
    spot, so we window around ATM before estimating the zero-crossing.
    """
    if strike_series is None or strike_series.empty:
        return None
    series = pd.Series(
        pd.to_numeric(strike_series, errors="coerce"),
        index=pd.to_numeric(strike_series.index, errors="coerce"),
    )
    series = series.dropna()
    series = series[~series.index.isna()].sort_index()
    if series.index.duplicated().any():
        series = series.groupby(level=0).sum()
    if len(series) < 2:
        return None

    spot_val = safe_float(spot, 0.0)
    local = series
    if spot_val > 0:
        windowed = select_atm_strike_series(
            series,
            spot_val,
            window_pct=window_pct,
            min_strikes=min_strikes,
        )
        if len(windowed) >= 2:
            local = windowed
    return estimate_gamma_flip(local.cumsum())


def greek_gamma_series_from_df(greek_df: pd.DataFrame | None) -> pd.Series:
    """Net gamma by strike from a UW greek-exposure DataFrame."""
    if greek_df is None or greek_df.empty:
        return pd.Series(dtype=float)
    df = greek_df.set_index("strike") if "strike" in greek_df.columns else greek_df
    if "net_gex" in df.columns:
        return pd.to_numeric(df["net_gex"], errors="coerce").dropna()
    if "GEX" in df.columns:
        return pd.to_numeric(df["GEX"], errors="coerce").dropna()
    if "call_gex" in df.columns and "put_gex" in df.columns:
        return pd.to_numeric(df["call_gex"], errors="coerce").fillna(0.0) + pd.to_numeric(
            df["put_gex"], errors="coerce"
        ).fillna(0.0)
    return pd.Series(dtype=float)


def magnet_gamma_from_call_put(greek_df: pd.DataFrame | None, spot: float | None) -> pd.Series:
    """Magnet profile: when call/put gamma cancel, show the dominant leg near spot."""
    if greek_df is None or greek_df.empty:
        return pd.Series(dtype=float)
    df = greek_df.set_index("strike") if "strike" in greek_df.columns else greek_df
    if "call_gex" not in df.columns or "put_gex" not in df.columns:
        return greek_gamma_series_from_df(greek_df)

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


def gamma_flip_from_uw_greek(
    greek_df: pd.DataFrame | None,
    spot: float | None,
    *,
    gex_by_strike: pd.Series | None = None,
    cumulative_gex: pd.Series | None = None,
) -> float | None:
    """ATM gamma flip from call/put surface (magnet profile when legs cancel)."""
    flip_series = magnet_gamma_from_call_put(greek_df, spot)
    if flip_series.empty and gex_by_strike is not None and not gex_by_strike.empty:
        flip_series = pd.Series(gex_by_strike, dtype=float).sort_index()
    return gamma_flip_from_profile(flip_series, spot)


def strike_center_of_mass(strike: pd.Series, spot: float | None = None) -> float:
    if strike.empty:
        return 0.0
    weights = strike.abs().values
    total = weights.sum()
    if total <= 0:
        return safe_float(spot, 0.0)
    return float(np.average(strike.index.astype(float), weights=weights))


def top_strike_concentration(strike: pd.Series, top_n: int = 5) -> float:
    if strike.empty:
        return 0.0
    total_abs = strike.abs().sum()
    if total_abs == 0:
        return 0.0
    top = strike.abs().sort_values(ascending=False).head(top_n).sum()
    return float(top / total_abs)


def term_structure_breakdown(
    expirations: pd.Series,
    *,
    snapshot_date: pd.Timestamp | None = None,
    near_term_buckets: int = 3,
) -> dict[str, float]:
    """Summarize expiration GEX into 0DTE/near-term/back-term buckets.

    When expiration dates are parseable and a snapshot date is known, 0DTE is
    the same-date expiration. Otherwise the first available bucket acts as a
    conservative same-day proxy for export files that only preserve ordering.
    """
    if expirations is None or expirations.empty:
        return {
            "term_total_gex_bn": 0.0,
            "zero_dte_gex_bn": 0.0,
            "zero_dte_ratio": 0.0,
            "front_term_gex_bn": 0.0,
            "front_term_ratio": 0.0,
            "near_term_gex_bn": 0.0,
            "near_term_ratio": 0.0,
            "back_term_gex_bn": 0.0,
            "back_term_ratio": 0.0,
            "term_curvature": 0.0,
            "expiration_count": 0.0,
        }

    exp = pd.Series(expirations, dtype=float).sort_index()
    total = float(exp.sum())
    abs_total = float(exp.abs().sum())

    zero_dte = 0.0
    if snapshot_date is not None:
        idx_dates = pd.to_datetime(exp.index, errors="coerce")
        valid_dates = pd.Series(idx_dates, index=exp.index).dt.date
        snap_date = pd.Timestamp(snapshot_date).date()
        same_day = exp.loc[valid_dates == snap_date]
        if not same_day.empty:
            zero_dte = float(same_day.sum())
    if zero_dte == 0.0 and not exp.empty:
        zero_dte = float(exp.iloc[0])

    near = float(exp.head(max(1, near_term_buckets)).sum())
    back = float(exp.tail(max(1, near_term_buckets)).sum())
    denom = total if total != 0 else abs_total

    # Front-term = net GEX of the single nearest-dated expiration. This is
    # distinct from ``zero_dte`` (same calendar-day expirations only): on days
    # without a same-day expiry the two diverge, so it carries independent
    # signal in the feature vector.
    front_term = float(exp.iloc[0]) if not exp.empty else 0.0

    return {
        "term_total_gex_bn": total,
        "zero_dte_gex_bn": zero_dte,
        "zero_dte_ratio": zero_dte / denom if denom else 0.0,
        "front_term_gex_bn": front_term,
        "front_term_ratio": front_term / denom if denom else 0.0,
        "near_term_gex_bn": near,
        "near_term_ratio": near / denom if denom else 0.0,
        "back_term_gex_bn": back,
        "back_term_ratio": back / denom if denom else 0.0,
        "term_curvature": near - back,
        "expiration_count": float(len(exp)),
    }


def cumulative_slope_at_spot(cumulative: pd.Series, spot: float) -> float:
    if cumulative.empty or spot <= 0:
        return 0.0
    idx = pd.to_numeric(cumulative.index, errors="coerce").astype(float)
    vals = cumulative.astype(float).values
    valid = ~np.isnan(idx)
    if int(np.sum(valid)) < 2:
        return 0.0
    x = idx[valid].values
    y = vals[valid]
    order = np.argsort(x)
    x, y = x[order], y[order]
    pos = np.searchsorted(x, spot)
    if pos <= 0:
        return float((y[1] - y[0]) / max(x[1] - x[0], 1e-9))
    if pos >= len(x):
        return float((y[-1] - y[-2]) / max(x[-1] - x[-2], 1e-9))
    x0, x1 = float(x[pos - 1]), float(x[pos])
    y0, y1 = float(y[pos - 1]), float(y[pos])
    return float((y1 - y0) / max(x1 - x0, 1e-9))


def select_atm_strike_series(
    series: pd.Series,
    spot: float | None,
    *,
    window_pct: float = 0.04,
    min_strikes: int = 5,
    max_strikes: int | None = None,
) -> pd.Series:
    """Keep strikes near ATM, expanding to nearest strikes when the window is sparse."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    cleaned = pd.Series(pd.to_numeric(series, errors="coerce"), index=pd.to_numeric(series.index, errors="coerce"))
    cleaned = cleaned.dropna()
    cleaned = cleaned[~cleaned.index.isna()]
    if cleaned.empty:
        return pd.Series(dtype=float)
    cleaned = cleaned.sort_index()
    if cleaned.index.duplicated().any():
        cleaned = cleaned.groupby(level=0).sum()

    spot_val = safe_float(spot, 0.0)
    target = max(min_strikes, max_strikes or len(cleaned))
    if spot_val > 0:
        lo, hi = spot_val * (1 - window_pct), spot_val * (1 + window_pct)
        window = cleaned.loc[(cleaned.index >= lo) & (cleaned.index <= hi)]
        if len(window) < min_strikes:
            distances = pd.Series(
                np.abs(cleaned.index.astype(float) - spot_val),
                index=cleaned.index,
            )
            nearest = distances.nsmallest(min(len(cleaned), target)).index
            window = cleaned.loc[nearest]
        cleaned = window.sort_index()

    if max_strikes and len(cleaned) > max_strikes and spot_val > 0:
        distances = pd.Series(
            np.abs(cleaned.index.astype(float) - spot_val),
            index=cleaned.index,
        )
        near = cleaned.loc[distances.nsmallest(max(max_strikes // 2, min_strikes)).index]
        peaks = cleaned.loc[cleaned.abs().nlargest(max(max_strikes // 2, min_strikes)).index]
        cleaned = cleaned.loc[near.index.union(peaks.index)].sort_index()
        if len(cleaned) > max_strikes:
            distances = pd.Series(
                np.abs(cleaned.index.astype(float) - spot_val),
                index=cleaned.index,
            )
            cleaned = cleaned.loc[distances.nsmallest(max_strikes).index].sort_index()

    return cleaned.sort_index()


def select_dense_atm_strike_series(
    series: pd.Series,
    spot: float | None,
    *,
    window_pct: float = 0.025,
    min_strikes: int = 8,
    max_strikes: int = 65,
) -> pd.Series:
    """Keep every strike inside the ATM band — trim only from the edges, never skip peaks."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    cleaned = pd.Series(pd.to_numeric(series, errors="coerce"), index=pd.to_numeric(series.index, errors="coerce"))
    cleaned = cleaned.dropna()
    cleaned = cleaned[~cleaned.index.isna()]
    if cleaned.empty:
        return pd.Series(dtype=float)
    cleaned = cleaned.sort_index()
    if cleaned.index.duplicated().any():
        cleaned = cleaned.groupby(level=0).sum()

    spot_val = safe_float(spot, 0.0)
    if spot_val <= 0:
        return cleaned.head(max_strikes)

    lo, hi = spot_val * (1 - window_pct), spot_val * (1 + window_pct)
    window = cleaned.loc[(cleaned.index >= lo) & (cleaned.index <= hi)]
    if len(window) < min_strikes:
        distances = pd.Series(
            np.abs(cleaned.index.astype(float) - spot_val),
            index=cleaned.index,
        )
        window = cleaned.loc[distances.nsmallest(min(len(cleaned), max_strikes)).index]

    if len(window) > max_strikes:
        distances = pd.Series(
            np.abs(window.index.astype(float) - spot_val),
            index=window.index,
        )
        window = window.loc[distances.nsmallest(max_strikes).index]

    return window.sort_index()


def _strike_on_grid(strike: float, grid: float) -> bool:
    """True when strike sits on a regular SPX-style grid (5, 10, 25, …)."""
    if grid <= 0:
        return False
    aligned = round(float(strike) / grid) * grid
    return abs(float(strike) - aligned) < 0.51


def snap_strike_grid_for_chart(
    series: pd.Series,
    spot: float | None,
    *,
    max_strikes: int,
    pin_strikes: tuple[float, ...] = (),
) -> pd.Series:
    """Keep a regular strike grid (5/10/25 pt) so chart bars line up cleanly."""
    if series is None or series.empty:
        return pd.Series(dtype=float)

    window = series.sort_index()
    if len(window) <= max(1, int(max_strikes)):
        return window

    spot_val = safe_float(spot, 0.0)
    pinned: set[float] = set()
    for raw in pin_strikes:
        try:
            target = float(raw)
        except (TypeError, ValueError):
            continue
        if target <= 0:
            continue
        idx_vals = window.index.astype(float)
        pinned.add(float(window.index[np.abs(idx_vals - target).argmin()]))
    if spot_val > 0:
        idx_vals = window.index.astype(float)
        pinned.add(float(window.index[np.abs(idx_vals - spot_val).argmin()]))
    for idx in window.abs().nlargest(min(3, len(window))).index:
        pinned.add(float(idx))

    chosen: pd.Series | None = None
    for grid in (5.0, 10.0, 25.0, 50.0):
        keys = [
            idx
            for idx in window.index
            if _strike_on_grid(float(idx), grid) or float(idx) in pinned
        ]
        if not keys:
            continue
        subset = window.loc[sorted(set(keys))].sort_index()
        if len(subset) <= max_strikes:
            chosen = subset
            break
        if chosen is None or len(subset) < len(chosen):
            chosen = subset

    if chosen is None or chosen.empty:
        chosen = window

    if len(chosen) <= max_strikes:
        return chosen.sort_index()

    keep_pinned = [idx for idx in chosen.index if float(idx) in pinned]
    droppable = [idx for idx in chosen.index if float(idx) not in pinned]
    if spot_val > 0 and droppable:
        distances = pd.Series(
            np.abs(pd.Series(droppable, dtype=float) - spot_val),
            index=droppable,
        )
        drop_n = len(chosen) - max_strikes
        drop = set(distances.nlargest(drop_n).index)
        keep = [idx for idx in chosen.index if idx in keep_pinned or idx not in drop]
        return chosen.loc[keep].sort_index()
    return chosen.iloc[:max_strikes].sort_index()


def spot_covers_strike_grid(series: pd.Series, spot: float, *, tolerance_pct: float = 0.015) -> bool:
    """True when spot lies inside the strike index range (with a small margin)."""
    if series is None or series.empty or spot <= 0:
        return False
    idx = pd.to_numeric(series.index, errors="coerce").dropna()
    if idx.empty:
        return False
    margin = spot * tolerance_pct
    return float(idx.min()) - margin <= spot <= float(idx.max()) + margin


def extract_surface_vector(
    strike: pd.Series,
    spot: float | None = None,
    window_pct: float = 0.05,
    n_bins: int = 32,
) -> np.ndarray:
    """Normalized strike GEX vector near spot for cosine similarity."""
    if strike.empty:
        return np.zeros(n_bins, dtype=float)
    spot = safe_float(spot, float(np.median(strike.index.astype(float))) if len(strike) else 0.0)
    near = select_atm_strike_series(strike, spot, window_pct=window_pct, min_strikes=5)
    if near.empty:
        near = strike
    lower = spot * (1 - window_pct)
    upper = spot * (1 + window_pct)
    edges = np.linspace(lower, upper, n_bins + 1)
    bins = np.zeros(n_bins, dtype=float)
    strikes = near.index.astype(float).values
    vals = near.values.astype(float)
    for s, v in zip(strikes, vals):
        bi = int(np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1))
        bins[bi] += v
    norm = np.linalg.norm(bins)
    return bins / norm if norm > 1e-12 else bins


def adaptive_surface_window(realized_vol: float, base_pct: float = 0.05) -> float:
    """Widen the strike window in higher-volatility regimes.

    In calm regimes the relevant gamma clusters tightly around spot; in stressed
    regimes far-OTM gamma matters more, so the cosine-similarity window should
    expand. Clamped to a sane [base, 3x] range.
    """
    rv = safe_float(realized_vol, 0.0)
    if rv <= 0:
        return base_pct
    # Map per-step realized vol (~0.002-0.02 typical for indices) onto a widening
    # multiplier of roughly 1x-3x.
    scale = 1.0 + min(2.0, rv * 100.0)
    return float(min(base_pct * 3.0, base_pct * scale))


def compute_features_from_exports(
    info: dict[str, Path],
    spot: float | None = None,
    prev_features: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build scalar feature dict from export file paths."""
    features: dict[str, float] = {}
    strike = pd.Series(dtype=float)
    cumulative = pd.Series(dtype=float)

    if "gex_by_strike" in info:
        strike = load_strike_series(info["gex_by_strike"])
        vals = strike.astype(float)
        features["total_gex_bn"] = float(vals.sum())
        features["pos_gex_bn"] = float(vals[vals > 0].sum())
        features["neg_gex_bn"] = float(vals[vals < 0].sum())
        features["gex_mean_bn"] = float(vals.mean()) if len(vals) else 0.0
        features["gex_std_bn"] = float(vals.std()) if len(vals) > 1 else 0.0
        mag = vals.abs().sort_values(ascending=False)
        for i in range(5):
            features[f"top_gex_{i + 1}"] = float(mag.iloc[i]) if i < len(mag) else 0.0
        features["call_wall"] = float(vals.idxmax()) if len(vals) else 0.0
        features["put_wall"] = float(vals.idxmin()) if len(vals) else 0.0
        features["wall_spread"] = features["call_wall"] - features["put_wall"]
        features["gex_concentration"] = top_strike_concentration(vals)
        features["gex_com"] = strike_center_of_mass(vals, spot)
    else:
        for key in (
            "total_gex_bn", "pos_gex_bn", "neg_gex_bn", "gex_mean_bn", "gex_std_bn",
            "call_wall", "put_wall", "wall_spread", "gex_concentration", "gex_com",
        ):
            features[key] = 0.0
        for i in range(5):
            features[f"top_gex_{i + 1}"] = 0.0

    if "gex_by_expiration" in info:
        exp = load_expiration_series(info["gex_by_expiration"])
        features.update(term_structure_breakdown(exp))
    else:
        features.update(term_structure_breakdown(pd.Series(dtype=float)))

    if spot is None and not strike.empty:
        spot = float(np.median(strike.index.astype(float)))
    features["spot"] = safe_float(spot, 0.0)

    greek_df = None
    if "greek_exposure" in info:
        from gex_core.exports import load_greek_exposure_df

        greek_df = load_greek_exposure_df(info["greek_exposure"])
    if "cumulative_gex" in info:
        cumulative = load_cumulative_series(info["cumulative_gex"])
    flip = resolve_gamma_flip(
        spot=features["spot"] if features["spot"] > 0 else None,
        gex_by_strike=strike if len(strike) else None,
        cumulative_gex=cumulative if len(cumulative) else None,
        greek_exposure_df=greek_df if greek_df is not None and not greek_df.empty else None,
    )
    features["gamma_flip"] = safe_float(flip, 0.0)

    if not cumulative.empty and features["spot"] > 0:
        features["flip_distance_pct"] = (
            (features["gamma_flip"] - features["spot"]) / features["spot"]
            if features["gamma_flip"]
            else 0.0
        )
        features["cum_slope_at_spot"] = cumulative_slope_at_spot(cumulative, features["spot"])
    else:
        features["flip_distance_pct"] = 0.0
        features["cum_slope_at_spot"] = 0.0

    if "gex_surface" in info:
        surface = load_surface_df(info["gex_surface"])
        if not surface.empty and "GEX" in surface.columns:
            g = surface["GEX"].astype(float)
            features["surface_mean_m"] = float(g.mean())
            features["surface_std_m"] = float(g.std()) if len(g) > 1 else 0.0
            features["surface_max_m"] = float(g.abs().max())
            features["surface_peak"] = features["surface_max_m"]
        else:
            features.update(
                {"surface_mean_m": 0.0, "surface_std_m": 0.0, "surface_max_m": 0.0, "surface_peak": 0.0}
            )
    else:
        features.update(
            {"surface_mean_m": 0.0, "surface_std_m": 0.0, "surface_max_m": 0.0, "surface_peak": 0.0}
        )

    if prev_features:
        features["total_gex_momentum"] = features["total_gex_bn"] - prev_features.get("total_gex_bn", 0.0)
        features["flip_velocity"] = features["gamma_flip"] - prev_features.get("gamma_flip", 0.0)
        features["near_term_ratio_delta"] = features["near_term_ratio"] - prev_features.get("near_term_ratio", 0.0)
        features["zero_dte_ratio_delta"] = features["zero_dte_ratio"] - prev_features.get("zero_dte_ratio", 0.0)
        features["term_curvature_delta"] = features["term_curvature"] - prev_features.get("term_curvature", 0.0)
    else:
        features["total_gex_momentum"] = 0.0
        features["flip_velocity"] = 0.0
        features["near_term_ratio_delta"] = 0.0
        features["zero_dte_ratio_delta"] = 0.0
        features["term_curvature_delta"] = 0.0

    market_date = None
    if "summary" in info:
        import json

        with info["summary"].open(encoding="utf-8") as f:
            summary = json.load(f)
        market_date = summary.get("market_date")
        extended = summary.get("extended_features") or {}
        features.update({k: safe_float(v) for k, v in extended.items()})
    features.update(event_calendar_features(market_date))
    if market_date:
        features["market_date"] = market_date

    for key, default in EXTENDED_FEATURE_DEFAULTS.items():
        features.setdefault(key, default)
    return features


def snapshot_feature_vector(row: dict[str, Any]) -> np.ndarray:
    """Scalar feature vector for KNN from a snapshot metrics dict."""
    base = [
        row["total_gex"],
        row["pos_gex"],
        row["neg_gex"],
        row["gex_std"],
        row["near_term_ratio"],
        row.get("surface_peak", 0.0),
        safe_float(row.get("call_wall"), 0.0),
        safe_float(row.get("put_wall"), 0.0),
        safe_float(row.get("gamma_flip"), 0.0),
        safe_float(row.get("wall_spread"), safe_float(row.get("call_wall"), 0.0) - safe_float(row.get("put_wall"), 0.0)),
        safe_float(row.get("flip_distance_pct"), 0.0),
        safe_float(row.get("total_gex_momentum"), 0.0),
        safe_float(row.get("flip_velocity"), 0.0),
        safe_float(row.get("gex_concentration"), 0.0),
        safe_float(row.get("cum_slope_at_spot"), 0.0),
        safe_float(row.get("zero_dte_ratio"), 0.0),
        safe_float(row.get("back_term_ratio"), 0.0),
        safe_float(row.get("term_curvature"), 0.0),
        safe_float(row.get("expiration_count"), 0.0),
        safe_float(row.get("realized_vol"), 0.0),
        safe_float(row.get("spot_return"), 0.0),
        safe_float(row.get("front_term_ratio"), 0.0),
    ]
    extended = [safe_float(row.get(name), 0.0) for name in extended_feature_names()]
    return np.array(base + extended, dtype=float)


def enrich_snapshot_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add derived prediction features to a snapshot metrics dict."""
    strike = metrics.get("strike", pd.Series(dtype=float))
    cumulative = metrics.get("cumulative", pd.Series(dtype=float))
    spot = safe_float(metrics.get("spot"), float(np.median(strike.index.astype(float))) if len(strike) else 0.0)
    call_wall = safe_float(metrics.get("call_wall"), 0.0)
    put_wall = safe_float(metrics.get("put_wall"), 0.0)
    greek_strike = metrics.get("greek_strike")
    greek_df = metrics.get("greek_exposure_df")
    spot_df = metrics.get("spot_exposures_df")
    if not isinstance(greek_df, pd.DataFrame):
        greek_df = None
    if not isinstance(spot_df, pd.DataFrame):
        spot_df = None
    strike_for_flip = pd.Series(strike, dtype=float).sort_index() if len(strike) else pd.Series(dtype=float)
    gamma_flip = resolve_gamma_flip(
        spot=spot if spot > 0 else None,
        gex_by_strike=strike_for_flip if len(strike_for_flip) else None,
        cumulative_gex=pd.Series(cumulative, dtype=float).sort_index() if len(cumulative) else None,
        greek_exposure_df=greek_df,
        greek_strike=greek_strike if isinstance(greek_strike, pd.Series) else None,
        spot_exposure_df=spot_df,
    )
    if gamma_flip is None:
        gamma_flip = parse_gamma_flip_value(metrics.get("gamma_flip"))

    metrics["wall_spread"] = call_wall - put_wall
    metrics["gex_concentration"] = top_strike_concentration(strike) if len(strike) else 0.0
    metrics["gex_com"] = strike_center_of_mass(strike, spot) if len(strike) else spot
    metrics["gamma_flip"] = gamma_flip
    metrics["flip_distance_pct"] = (
        (safe_float(gamma_flip) - spot) / spot if gamma_flip is not None and spot > 0 else 0.0
    )
    metrics["cum_slope_at_spot"] = cumulative_slope_at_spot(cumulative, spot) if len(cumulative) and spot > 0 else 0.0
    window_pct = adaptive_surface_window(metrics.get("realized_vol", 0.0))
    metrics["surface_vector"] = extract_surface_vector(strike, spot, window_pct=window_pct)
    metrics["spot"] = spot
    for key in (
        "term_total_gex_bn",
        "zero_dte_gex_bn",
        "zero_dte_ratio",
        "front_term_gex_bn",
        "front_term_ratio",
        "near_term_gex_bn",
        "near_term_ratio",
        "back_term_gex_bn",
        "back_term_ratio",
        "term_curvature",
        "expiration_count",
        "zero_dte_ratio_delta",
        "term_curvature_delta",
    ):
        metrics[key] = safe_float(metrics.get(key), 0.0)
    return apply_extended_defaults(metrics)
