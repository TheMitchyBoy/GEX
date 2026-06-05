"""Market-context features (realized volatility, returns, optional VIX).

Realized volatility and spot returns are derived purely from the snapshot
``spot`` history, so they work fully offline. VIX is optional and fetched via
``yfinance`` when available; any failure (no network, missing dependency)
degrades silently to ``0.0`` so forecasting never hard-fails in a sandbox.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from gex_core.features import safe_float

logger = logging.getLogger(__name__)

# Annualization assuming ~252 trading days; snapshot cadence is treated as
# one step. Callers that need calendar scaling can post-multiply.
TRADING_PERIODS = 252


def spot_log_returns(history: list[dict[str, Any]]) -> list[float]:
    """Log returns of the snapshot spot series (skips missing/zero spots)."""
    returns: list[float] = []
    prev: float | None = None
    for row in history:
        spot = safe_float(row.get("spot"), 0.0)
        if spot <= 0:
            prev = None
            continue
        if prev is not None and prev > 0:
            returns.append(math.log(spot / prev))
        prev = spot
    return returns


def realized_volatility(history: list[dict[str, Any]], window: int = 10) -> float:
    """Rolling realized vol (std of recent log returns).

    Returns the *per-step* standard deviation; this is a stationary, scale-free
    feature that captures the current volatility regime without any network
    dependency.
    """
    returns = spot_log_returns(history)
    if len(returns) < 2:
        return 0.0
    recent = returns[-window:]
    if len(recent) < 2:
        return 0.0
    return float(np.std(recent, ddof=1))


def annualized_realized_vol(history: list[dict[str, Any]], window: int = 10) -> float:
    return realized_volatility(history, window=window) * math.sqrt(TRADING_PERIODS)


def latest_spot_return(history: list[dict[str, Any]]) -> float:
    """Most recent step's simple spot return."""
    if len(history) < 2:
        return 0.0
    s0 = safe_float(history[-2].get("spot"), 0.0)
    s1 = safe_float(history[-1].get("spot"), 0.0)
    if s0 <= 0 or s1 <= 0:
        return 0.0
    return (s1 - s0) / s0


def fetch_vix_level() -> float:
    """Best-effort current VIX level; ``0.0`` when unavailable (offline)."""
    try:
        import yfinance as yf

        data = yf.Ticker("^VIX").history(period="5d")
        if data is None or data.empty:
            return 0.0
        return float(data["Close"].dropna().iloc[-1])
    except Exception as exc:  # pragma: no cover - network/optional dep path
        logger.debug("VIX fetch unavailable: %s", exc)
        return 0.0


def attach_market_features(
    history: list[dict[str, Any]],
    *,
    vol_window: int = 10,
    include_vix: bool = False,
) -> list[dict[str, Any]]:
    """Annotate each snapshot in-place with realized-vol / return features.

    ``realized_vol`` and ``spot_return`` are computed from the trailing window
    up to and including that snapshot, so the features are causal (no
    lookahead) and safe to use as model inputs.
    """
    vix = fetch_vix_level() if include_vix else 0.0
    for i, row in enumerate(history):
        trailing = history[: i + 1]
        row["realized_vol"] = realized_volatility(trailing, window=vol_window)
        row["spot_return"] = latest_spot_return(trailing)
        row["vix_level"] = vix
    return history
