"""Feature engineering for GEX prediction and similarity matching."""

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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
    lower = spot * (1 - window_pct)
    upper = spot * (1 + window_pct)
    near = strike.loc[(strike.index >= lower) & (strike.index <= upper)]
    if near.empty:
        near = strike
    edges = np.linspace(lower, upper, n_bins + 1)
    bins = np.zeros(n_bins, dtype=float)
    strikes = near.index.astype(float).values
    vals = near.values.astype(float)
    for s, v in zip(strikes, vals):
        bi = int(np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1))
        bins[bi] += v
    norm = np.linalg.norm(bins)
    return bins / norm if norm > 1e-12 else bins


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
        features["term_total_gex_bn"] = float(exp.sum())
        features["near_term_gex_bn"] = float(exp.head(3).sum()) if len(exp) else 0.0
        features["near_term_ratio"] = (
            features["near_term_gex_bn"] / features["term_total_gex_bn"]
            if features["term_total_gex_bn"] != 0
            else 0.0
        )
        features["back_term_gex_bn"] = float(exp.tail(3).sum()) if len(exp) else 0.0
        features["term_curvature"] = features["near_term_gex_bn"] - features["back_term_gex_bn"]
    else:
        features.update(
            {
                "term_total_gex_bn": 0.0,
                "near_term_gex_bn": 0.0,
                "near_term_ratio": 0.0,
                "back_term_gex_bn": 0.0,
                "term_curvature": 0.0,
            }
        )

    if "cumulative_gex" in info:
        cumulative = load_cumulative_series(info["cumulative_gex"])
        flip = estimate_gamma_flip(cumulative)
        features["gamma_flip"] = safe_float(flip, 0.0)
    else:
        features["gamma_flip"] = 0.0

    if spot is None and not strike.empty:
        spot = float(np.median(strike.index.astype(float)))
    features["spot"] = safe_float(spot, 0.0)

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
    else:
        features["total_gex_momentum"] = 0.0
        features["flip_velocity"] = 0.0
        features["near_term_ratio_delta"] = 0.0

    return features


def snapshot_feature_vector(row: dict[str, Any]) -> np.ndarray:
    """Scalar feature vector for KNN from a snapshot metrics dict."""
    return np.array(
        [
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
        ],
        dtype=float,
    )


def enrich_snapshot_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add derived prediction features to a snapshot metrics dict."""
    strike = metrics.get("strike", pd.Series(dtype=float))
    cumulative = metrics.get("cumulative", pd.Series(dtype=float))
    spot = safe_float(metrics.get("spot"), float(np.median(strike.index.astype(float))) if len(strike) else 0.0)
    call_wall = safe_float(metrics.get("call_wall"), 0.0)
    put_wall = safe_float(metrics.get("put_wall"), 0.0)
    gamma_flip = metrics.get("gamma_flip")

    metrics["wall_spread"] = call_wall - put_wall
    metrics["gex_concentration"] = top_strike_concentration(strike) if len(strike) else 0.0
    metrics["gex_com"] = strike_center_of_mass(strike, spot) if len(strike) else spot
    metrics["flip_distance_pct"] = (
        (safe_float(gamma_flip) - spot) / spot if gamma_flip is not None and spot > 0 else 0.0
    )
    metrics["cum_slope_at_spot"] = cumulative_slope_at_spot(cumulative, spot) if len(cumulative) and spot > 0 else 0.0
    metrics["surface_vector"] = extract_surface_vector(strike, spot)
    metrics["spot"] = spot
    return metrics
