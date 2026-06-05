"""Structural attribution of GEX changes from snapshot fields alone.

This bridges the gap noted in the forecasting review: ``decompose.py`` needs a
cached raw option chain, so it was never wired into the live forecast. Here we
attribute the *last observed* ΔGEX into a spot-driven component and a residual
(time/vol/flow/repositioning) using only fields already present on each
snapshot, so it runs anywhere the dashboard runs.
"""

from __future__ import annotations

from typing import Any

from gex_core.features import safe_float


def attribute_last_move(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Split the most recent ΔGEX into spot vs residual contributions.

    The spot component uses the cumulative-GEX slope at spot as a local
    sensitivity ``d(GEX)/d(strike)``: a spot move of ``Δspot`` points implies a
    first-order GEX change of ``slope * Δspot``. Whatever is left over is
    attributed to time decay, volatility, and flow/repositioning.
    """
    if len(history) < 2:
        return None
    prev, cur = history[-2], history[-1]
    prev_total = safe_float(prev.get("total_gex"), 0.0)
    cur_total = safe_float(cur.get("total_gex"), 0.0)
    observed = cur_total - prev_total

    prev_spot = safe_float(prev.get("spot"), 0.0)
    cur_spot = safe_float(cur.get("spot"), 0.0)
    slope = safe_float(prev.get("cum_slope_at_spot"), 0.0)
    spot_component = 0.0
    if prev_spot > 0 and cur_spot > 0:
        spot_component = slope * (cur_spot - prev_spot)

    residual = observed - spot_component
    return {
        "observed_delta_gex": observed,
        "spot_component": spot_component,
        "residual_component": residual,
        "spot_move_pts": (cur_spot - prev_spot) if (prev_spot > 0 and cur_spot > 0) else None,
    }


def structural_forward_delta(history: list[dict[str, Any]], decay: float = 0.5) -> float | None:
    """Exponentially-weighted estimate of the next ΔGEX from recent attribution.

    Uses the residual (non-spot) part of recent moves as a slow-moving
    structural drift baseline, discounting older observations. Returns ``None``
    when there is not enough history.
    """
    if len(history) < 3:
        return None
    residuals: list[float] = []
    for i in range(1, len(history)):
        window = history[: i + 1]
        attr = attribute_last_move(window)
        if attr is not None:
            residuals.append(attr["residual_component"])
    if not residuals:
        return None
    weight = 1.0
    weighted_sum = 0.0
    norm = 0.0
    for value in reversed(residuals):
        weighted_sum += weight * value
        norm += weight
        weight *= decay
    return weighted_sum / norm if norm > 0 else None
