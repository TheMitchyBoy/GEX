"""Consensus spot price from multiple UW sources."""

from __future__ import annotations

import os
from typing import Any

from gex_core.features import safe_float


def spot_disagreement_tolerance_pct() -> float:
    try:
        return float(os.environ.get("GEX_SPOT_DISAGREEMENT_TOLERANCE_PCT", "0.005"))
    except (TypeError, ValueError):
        return 0.005


def build_spot_consensus(
    *,
    stock_state: float | None = None,
    spot_exposure_price: float | None = None,
    intraday_price: float | None = None,
    chosen: float | None = None,
) -> dict[str, Any]:
    """Compare UW spot candidates and record disagreement."""
    candidates: dict[str, float] = {}
    if stock_state and stock_state > 0:
        candidates["stock_state"] = float(stock_state)
    if spot_exposure_price and spot_exposure_price > 0:
        candidates["spot_exposures"] = float(spot_exposure_price)
    if intraday_price and intraday_price > 0:
        candidates["intraday"] = float(intraday_price)
    if chosen and chosen > 0:
        candidates["chosen"] = float(chosen)

    values = list(candidates.values())
    spot = safe_float(chosen, values[0] if values else 0.0)
    if spot <= 0 and values:
        spot = values[0]

    disagreement_pct = 0.0
    if len(values) >= 2 and spot > 0:
        disagreement_pct = (max(values) - min(values)) / spot

    source = "chosen"
    if spot > 0:
        for name, value in candidates.items():
            if abs(value - spot) < 1e-9:
                source = name
                break

    return {
        "spot": spot if spot > 0 else None,
        "spot_source": source,
        "spot_candidates": candidates,
        "spot_disagreement_pct": round(disagreement_pct, 6),
        "spot_disagreement": disagreement_pct > spot_disagreement_tolerance_pct(),
    }
