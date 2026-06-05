"""Lightweight market-regime detection for regime-conditional forecasting.

The previous pipeline only distinguished LONG vs SHORT gamma from the sign of
total GEX. This module adds a volatility dimension so blending and confidence
can adapt to calm vs stressed conditions without pulling in a heavy HMM.
"""

from __future__ import annotations

from typing import Any

from gex_core.features import safe_float
from gex_core.market_features import realized_volatility, spot_log_returns


def gamma_sign_regime(total_gex: float) -> str:
    return "LONG gamma" if safe_float(total_gex, 0.0) >= 0 else "SHORT gamma"


def volatility_bucket(history: list[dict[str, Any]], window: int = 10) -> str:
    """Classify the current volatility regime relative to recent history.

    Buckets the latest realized vol against the distribution of realized vol
    over the available history. Falls back to ``"unknown"`` when there is not
    enough spot data to estimate.
    """
    returns = spot_log_returns(history)
    if len(returns) < 4:
        return "unknown"
    current = realized_volatility(history, window=window)
    # Build a small distribution of trailing realized-vol estimates.
    samples = []
    for i in range(3, len(history)):
        samples.append(realized_volatility(history[: i + 1], window=window))
    samples = [s for s in samples if s > 0]
    if len(samples) < 3 or current <= 0:
        return "unknown"
    samples_sorted = sorted(samples)
    rank = sum(1 for s in samples_sorted if s <= current) / len(samples_sorted)
    if rank >= 0.66:
        return "high-vol"
    if rank <= 0.33:
        return "low-vol"
    return "mid-vol"


def classify_regime(history: list[dict[str, Any]], window: int = 10) -> dict[str, Any]:
    """Return a combined regime label for the most recent snapshot."""
    if not history:
        return {"gamma": "N/A", "volatility": "unknown", "label": "N/A"}
    latest = history[-1]
    gamma = gamma_sign_regime(latest.get("total_gex", 0.0))
    vol = volatility_bucket(history, window=window)
    return {
        "gamma": gamma,
        "volatility": vol,
        "label": f"{gamma} / {vol}",
    }


def model_blend_weight(volatility: str) -> float:
    """Regime-conditional weight for the trained-model overlay vs KNN.

    Trained regressors generalize better in calmer regimes; in high-vol regimes
    the instance-based KNN analog tends to react faster, so we lean on it more.
    """
    return {
        "low-vol": 0.6,
        "mid-vol": 0.5,
        "high-vol": 0.4,
        "unknown": 0.5,
    }.get(volatility, 0.5)
