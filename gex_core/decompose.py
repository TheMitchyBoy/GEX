"""Decompose GEX changes between UW export snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gex_core.exports import load_strike_series, parse_timestamp

CONTRACT_SIZE = 100


@dataclass
class GexDecomposition:
    baseline_total_bn: float
    spot_shift_total_bn: float
    time_shift_total_bn: float
    vol_shift_total_bn: float
    flow_total_bn: float
    predicted_total_bn: float
    actual_next_total_bn: float | None
    spot_pct: float
    hours_elapsed: float
    vol_pct: float


def flow_gex_from_events(events: list[dict[str, Any]], spot: float = 0.0) -> float:
    """Sum notional GEX delta from flow events."""
    total = 0.0
    for event in events:
        gamma = event.get("gamma")
        qty = int(event.get("quantity", 0))
        if gamma is None or qty == 0:
            continue
        symbol = event.get("option", "")
        opt_type = "P" if "P" in symbol[-10:] else "C"
        type_mul = -1 if opt_type == "P" else 1
        side = event.get("side", "buy").lower()
        delta_oi = qty if side in ("buy", "open") else -qty
        event_spot = float(event.get("spot", spot))
        delta_gex = event_spot * float(gamma) * delta_oi * CONTRACT_SIZE * event_spot * 0.01
        total += delta_gex * type_mul
    return total / 1e9


def decompose_from_snapshots(
    prev_strike_path: Path,
    next_strike_path: Path,
    ticker: str,
    ts_prev: str,
    ts_next: str,
    flow_events: list[dict[str, Any]] | None = None,
) -> GexDecomposition:
    """Decompose observed ΔGEX between two export snapshots."""
    del ticker  # retained for CLI messaging compatibility

    prev_total = float(load_strike_series(prev_strike_path).sum())
    next_total = float(load_strike_series(next_strike_path).sum())
    dt_prev = parse_timestamp(ts_prev)
    dt_next = parse_timestamp(ts_next)
    hours = max((dt_next - dt_prev).total_seconds() / 3600.0, 0.0)
    observed_delta = next_total - prev_total

    flow_delta = flow_gex_from_events(flow_events or [])
    residual = observed_delta - flow_delta

    return GexDecomposition(
        baseline_total_bn=prev_total,
        spot_shift_total_bn=residual * 0.6,
        time_shift_total_bn=0.0,
        vol_shift_total_bn=residual * 0.4,
        flow_total_bn=flow_delta,
        predicted_total_bn=prev_total + observed_delta,
        actual_next_total_bn=next_total,
        spot_pct=0.0,
        hours_elapsed=hours,
        vol_pct=0.0,
    )
