"""Decompose GEX changes into spot, time, vol, and flow components.

Prefer :func:`decompose_from_snapshots` — it compares two UW export snapshots
and is what the live forecast stack uses via :mod:`gex_core.structural`.

:func:`decompose_gex` remains for hypothetical what-if scenarios but needs a
legacy per-contract JSON cache at ``data/{TICKER}.json`` (CBOE-style payload).
The UW-only CLI no longer writes that file; use ``--compare-snapshots`` instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gex_core.pipeline import attach_signed_gex, parse_payload
from gex_core.data_quality import clean_option_data
from gex_core.exports import load_strike_series, parse_timestamp


DATA_DIR = Path("data")
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


def load_cached_options(ticker: str) -> tuple[float, pd.DataFrame]:
    cache_file = DATA_DIR / f"{ticker.upper()}.json"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"No per-contract cache at {cache_file}. "
            "Use scripts/gex_decompose.py --compare-snapshots for UW export snapshots."
        )
    with cache_file.open() as f:
        payload = json.load(f)
    spot, df = parse_payload(payload)
    df, _report = clean_option_data(df, spot=spot)
    if df is None or df.empty:
        raise ValueError(f"Cached options data for {ticker} is empty after cleaning.")
    return spot, df


def total_gex_bn(data: pd.DataFrame, spot: float) -> float:
    signed = attach_signed_gex(spot, data)
    return float(signed["GEX"].sum() / 1e9)


def decompose_spot_effect(data: pd.DataFrame, spot: float, spot_pct: float) -> float:
    """ΔGEX from repricing at shifted spot with fixed OI and gamma."""
    if spot_pct == 0:
        return 0.0
    new_spot = spot * (1 + spot_pct)
    base = total_gex_bn(data, spot)
    shifted = total_gex_bn(data, new_spot)
    return shifted - base


def decompose_time_effect(data: pd.DataFrame, spot: float, hours: float) -> float:
    """Approximate charm decay by shifting expiration dates forward."""
    if hours <= 0 or data.empty:
        return 0.0
    shifted = data.copy()
    delta = timedelta(hours=hours)
    shifted["expiration"] = shifted["expiration"] - delta
    today = datetime.today()
    shifted = shifted.loc[shifted["expiration"] > today]
    if shifted.empty:
        return -total_gex_bn(data, spot)
    base = total_gex_bn(data, spot)
    decayed = total_gex_bn(shifted, spot)
    return decayed - base


def decompose_vol_effect(data: pd.DataFrame, spot: float, vol_pct: float) -> float:
    """Approximate vol sensitivity by scaling gamma (proxy for ∂GEX/∂σ)."""
    if vol_pct == 0 or "gamma" not in data.columns:
        return 0.0
    scaled = data.copy()
    scaled["gamma"] = scaled["gamma"].astype(float) * (1 + vol_pct)
    base = total_gex_bn(data, spot)
    adjusted = total_gex_bn(scaled, spot)
    return adjusted - base


def flow_gex_from_events(events: list[dict[str, Any]], spot: float) -> float:
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


def decompose_gex(
    ticker: str,
    spot_pct: float = 0.0,
    hours_elapsed: float = 0.0,
    vol_pct: float = 0.0,
    flow_events: list[dict[str, Any]] | None = None,
    actual_next_total_bn: float | None = None,
) -> GexDecomposition:
    spot, data = load_cached_options(ticker)
    baseline = total_gex_bn(data, spot)
    spot_delta = decompose_spot_effect(data, spot, spot_pct)
    time_delta = decompose_time_effect(data, spot, hours_elapsed)
    vol_delta = decompose_vol_effect(data, spot, vol_pct)
    flow_delta = flow_gex_from_events(flow_events or [], spot)
    predicted = baseline + spot_delta + time_delta + vol_delta + flow_delta
    return GexDecomposition(
        baseline_total_bn=baseline,
        spot_shift_total_bn=spot_delta,
        time_shift_total_bn=time_delta,
        vol_shift_total_bn=vol_delta,
        flow_total_bn=flow_delta,
        predicted_total_bn=predicted,
        actual_next_total_bn=actual_next_total_bn,
        spot_pct=spot_pct,
        hours_elapsed=hours_elapsed,
        vol_pct=vol_pct,
    )


def decompose_from_snapshots(
    prev_strike_path: Path,
    next_strike_path: Path,
    ticker: str,
    ts_prev: str,
    ts_next: str,
    flow_events: list[dict[str, Any]] | None = None,
) -> GexDecomposition:
    """Decompose observed ΔGEX between two export snapshots."""
    prev_total = float(load_strike_series(prev_strike_path).sum())
    next_total = float(load_strike_series(next_strike_path).sum())
    dt_prev = parse_timestamp(ts_prev)
    dt_next = parse_timestamp(ts_next)
    hours = max((dt_next - dt_prev).total_seconds() / 3600.0, 0.0)
    observed_delta = next_total - prev_total

    try:
        spot, data = load_cached_options(ticker)
    except FileNotFoundError:
        return GexDecomposition(
            baseline_total_bn=prev_total,
            spot_shift_total_bn=0.0,
            time_shift_total_bn=0.0,
            vol_shift_total_bn=0.0,
            flow_total_bn=0.0,
            predicted_total_bn=prev_total,
            actual_next_total_bn=next_total,
            spot_pct=0.0,
            hours_elapsed=hours,
            vol_pct=0.0,
        )

    time_delta = decompose_time_effect(data, spot, hours)
    flow_delta = flow_gex_from_events(flow_events or [], spot)
    residual = observed_delta - time_delta - flow_delta

    return GexDecomposition(
        baseline_total_bn=prev_total,
        spot_shift_total_bn=residual * 0.6,
        time_shift_total_bn=time_delta,
        vol_shift_total_bn=residual * 0.4,
        flow_total_bn=flow_delta,
        predicted_total_bn=prev_total + time_delta + flow_delta + residual * 0.6 + residual * 0.4,
        actual_next_total_bn=next_total,
        spot_pct=0.0,
        hours_elapsed=hours,
        vol_pct=0.0,
    )
