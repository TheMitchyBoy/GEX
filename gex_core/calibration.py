"""Fitted, data-driven relationships that replace hard-coded forecast scalars.

Everything here degrades gracefully: when there is not enough history to fit a
stable relationship, the functions fall back to the previous hand-tuned
defaults so behaviour never gets *worse* than the original heuristics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gex_core.features import safe_float

# Legacy hand-tuned fallbacks (kept so sparse-history behaviour is unchanged).
DEFAULT_MOVE_PER_DELTA_GEX = 0.00035
MIN_SAMPLES_FOR_FIT = 6


def _spot_series(history: list[dict[str, Any]]) -> list[float]:
    return [safe_float(row.get("spot"), 0.0) for row in history]


def _forward_returns_and_deltas(
    history: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pair each snapshot's ΔGEX with the realized forward spot return."""
    deltas: list[float] = []
    returns: list[float] = []
    for i in range(len(history) - 1):
        s0 = safe_float(history[i].get("spot"), 0.0)
        s1 = safe_float(history[i + 1].get("spot"), 0.0)
        t0 = safe_float(history[i].get("total_gex"), 0.0)
        t1 = safe_float(history[i + 1].get("total_gex"), 0.0)
        if s0 <= 0 or s1 <= 0:
            continue
        deltas.append(t1 - t0)
        returns.append((s1 - s0) / s0)
    return np.asarray(deltas, dtype=float), np.asarray(returns, dtype=float)


def fit_move_per_delta_gex(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Least-squares slope of forward spot return on ΔGEX.

    Replaces the magic ``predicted_delta_gex * 0.00035`` constant. Returns the
    fitted slope plus diagnostics; falls back to the legacy default when the
    fit is unstable or under-sampled.
    """
    deltas, returns = _forward_returns_and_deltas(history)
    fitted = False
    slope = DEFAULT_MOVE_PER_DELTA_GEX
    r2 = None
    if len(deltas) >= MIN_SAMPLES_FOR_FIT and np.std(deltas) > 1e-9:
        # slope through the data; intercept folded out by centring.
        dx = deltas - deltas.mean()
        dy = returns - returns.mean()
        denom = float(np.dot(dx, dx))
        if denom > 1e-12:
            slope = float(np.dot(dx, dy) / denom)
            ss_tot = float(np.dot(dy, dy))
            ss_res = float(np.dot(dy - slope * dx, dy - slope * dx))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
            fitted = True
    return {
        "slope": slope,
        "fitted": fitted,
        "samples": int(len(deltas)),
        "r2": r2,
    }


def expected_directional_move_pct(
    predicted_delta_gex: float,
    history: list[dict[str, Any]],
) -> float:
    """Expected forward spot return implied by a forecast ΔGEX."""
    fit = fit_move_per_delta_gex(history)
    return safe_float(predicted_delta_gex, 0.0) * fit["slope"]


def fit_close_above_flip_rate(history: list[dict[str, Any]]) -> float | None:
    """Empirical base rate of 'next snapshot closes above its gamma flip'.

    Used to calibrate the geometric sigmoid in the forecast probabilities.
    Returns ``None`` when history is too short to estimate.
    """
    above = 0
    total = 0
    for row in history:
        spot = safe_float(row.get("spot"), 0.0)
        flip = safe_float(row.get("gamma_flip"), 0.0)
        if spot <= 0 or flip <= 0:
            continue
        total += 1
        if spot >= flip:
            above += 1
    if total < MIN_SAMPLES_FOR_FIT:
        return None
    return above / total


def calibrate_confidence(
    raw_confidence: float,
    hit_rate: float | None,
    n: int,
    *,
    prior_strength: float = 4.0,
) -> float:
    """Blend a raw distance-based confidence with empirical backtest hit-rate.

    The empirical hit-rate is shrunk toward 0.5 based on sample size ``n`` so
    that a handful of lucky backtest points cannot inflate confidence. When no
    backtest is available, the raw confidence is returned unchanged.
    """
    raw = max(0.0, min(1.0, safe_float(raw_confidence, 0.0)))
    if hit_rate is None or n <= 0:
        return raw
    # Shrink the observed hit-rate toward 0.5 (no-skill) with a beta-style prior.
    shrunk = (hit_rate * n + 0.5 * prior_strength) / (n + prior_strength)
    weight = n / (n + prior_strength)
    calibrated = (1.0 - weight) * raw + weight * shrunk
    return max(0.0, min(1.0, calibrated))
