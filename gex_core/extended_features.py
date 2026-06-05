"""Extended prediction features: greeks, flow, spot exposures, cross-asset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gex_core.event_calendar import event_calendar_features


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _top_strike_concentration(strike: pd.Series, top_n: int = 5) -> float:
    if strike.empty:
        return 0.0
    total_abs = strike.abs().sum()
    if total_abs == 0:
        return 0.0
    top = strike.abs().sort_values(ascending=False).head(top_n).sum()
    return float(top / total_abs)

EXTENDED_FEATURE_DEFAULTS: dict[str, float] = {
    "net_charm_bn": 0.0,
    "net_vanna_bn": 0.0,
    "net_delta_bn": 0.0,
    "charm_at_spot_bn": 0.0,
    "vanna_at_spot_bn": 0.0,
    "delta_at_spot_bn": 0.0,
    "charm_concentration": 0.0,
    "vanna_concentration": 0.0,
    "gamma_oi_bn": 0.0,
    "gamma_vol_bn": 0.0,
    "gamma_oi_vol_ratio": 0.0,
    "flow_event_count": 0.0,
    "flow_net_delta_gex_bn": 0.0,
    "flow_buy_ratio": 0.0,
    "flow_aggressiveness": 0.0,
    "vix_level": 0.0,
    "vix9d_level": 0.0,
    "iv_rank": 0.0,
    "skew_proxy": 0.0,
    "expected_move_pct": 0.0,
    "spy_return": 0.0,
    "tlt_return": 0.0,
    "is_fomc_week": 0.0,
    "is_cpi_day": 0.0,
    "is_nfp_day": 0.0,
    "is_opex_week": 0.0,
    "is_quad_witching": 0.0,
    "event_risk_score": 0.0,
}


def extended_feature_names() -> list[str]:
    return list(EXTENDED_FEATURE_DEFAULTS.keys())


def apply_extended_defaults(metrics: dict[str, Any]) -> dict[str, Any]:
    for key, default in EXTENDED_FEATURE_DEFAULTS.items():
        metrics[key] = _safe_float(metrics.get(key), default)
    return metrics


def _nearest_strike_value(series: pd.Series, spot: float) -> float:
    if series is None or series.empty or spot <= 0:
        return 0.0
    idx = pd.to_numeric(series.index, errors="coerce").astype(float)
    valid = series.copy()
    valid.index = idx
    valid = valid[~np.isnan(valid.index)]
    if valid.empty:
        return 0.0
    nearest = float(valid.index[np.argmin(np.abs(valid.index - spot))])
    return float(valid.loc[nearest])


def summarize_greek_exposure_df(df: pd.DataFrame, spot: float | None = None) -> dict[str, float]:
    """Aggregate charm/vanna/delta from UW greek-exposure strike table."""
    if df is None or df.empty:
        return {}
    spot = _safe_float(spot, float(pd.to_numeric(df["strike"], errors="coerce").median()) if "strike" in df.columns else 0.0)
    out: dict[str, float] = {}
    if "call_charm" in df.columns and "put_charm" in df.columns:
        charm = pd.to_numeric(df["call_charm"], errors="coerce").fillna(0.0) + pd.to_numeric(
            df["put_charm"], errors="coerce"
        ).fillna(0.0)
        out["net_charm_bn"] = float(charm.sum())
        out["charm_at_spot_bn"] = _nearest_strike_value(
            pd.Series(charm.values, index=df["strike"].values), spot
        )
        out["charm_concentration"] = _top_strike_concentration(pd.Series(charm.values, index=df["strike"].values))
    if "call_vanna" in df.columns and "put_vanna" in df.columns:
        vanna = pd.to_numeric(df["call_vanna"], errors="coerce").fillna(0.0) + pd.to_numeric(
            df["put_vanna"], errors="coerce"
        ).fillna(0.0)
        out["net_vanna_bn"] = float(vanna.sum())
        out["vanna_at_spot_bn"] = _nearest_strike_value(
            pd.Series(vanna.values, index=df["strike"].values), spot
        )
        out["vanna_concentration"] = _top_strike_concentration(pd.Series(vanna.values, index=df["strike"].values))
    if "call_delta" in df.columns and "put_delta" in df.columns:
        delta = pd.to_numeric(df["call_delta"], errors="coerce").fillna(0.0) + pd.to_numeric(
            df["put_delta"], errors="coerce"
        ).fillna(0.0)
        out["net_delta_bn"] = float(delta.sum())
        out["delta_at_spot_bn"] = _nearest_strike_value(
            pd.Series(delta.values, index=df["strike"].values), spot
        )
    return out


def summarize_spot_exposures_df(df: pd.DataFrame) -> dict[str, float]:
    """Summarize intraday gamma OI vs volume from UW spot-exposures."""
    if df is None or df.empty:
        return {}
    out: dict[str, float] = {}
    if "net_gamma_oi_bn" in df.columns:
        oi = pd.to_numeric(df["net_gamma_oi_bn"], errors="coerce").fillna(0.0)
        out["gamma_oi_bn"] = float(oi.sum())
    elif "call_gamma_oi" in df.columns and "put_gamma_oi" in df.columns:
        oi = pd.to_numeric(df["call_gamma_oi"], errors="coerce").fillna(0.0) + pd.to_numeric(
            df["put_gamma_oi"], errors="coerce"
        ).fillna(0.0)
        out["gamma_oi_bn"] = float(oi.sum()) / 1e9
    if "call_gamma_vol" in df.columns and "put_gamma_vol" in df.columns:
        vol = pd.to_numeric(df["call_gamma_vol"], errors="coerce").fillna(0.0) + pd.to_numeric(
            df["put_gamma_vol"], errors="coerce"
        ).fillna(0.0)
        out["gamma_vol_bn"] = float(vol.sum()) / 1e9
    oi_abs = abs(out.get("gamma_oi_bn", 0.0))
    vol_abs = abs(out.get("gamma_vol_bn", 0.0))
    out["gamma_oi_vol_ratio"] = vol_abs / oi_abs if oi_abs > 1e-9 else 0.0
    return out


def summarize_flow_feed(path: str | Path | None = None) -> dict[str, float]:
    """Aggregate recent option flow JSONL into scalar features."""
    feed = Path(path or os.environ.get("GEX_FLOW_FEED", "data/flow_sample.jsonl"))
    if not feed.is_file():
        return {}
    events: list[dict[str, Any]] = []
    for line in feed.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not events:
        return {}
    buys = sum(1 for e in events if str(e.get("side", "")).lower() == "buy")
    total_qty = sum(abs(int(e.get("quantity", 0) or 0)) for e in events)
    delta_gex = 0.0
    for event in events:
        gamma = _safe_float(event.get("gamma"), 0.0)
        qty = abs(int(event.get("quantity", 0) or 0))
        spot = _safe_float(event.get("spot"), 0.0)
        if gamma <= 0 or qty <= 0 or spot <= 0:
            continue
        sign = 1.0 if str(event.get("side", "buy")).lower() == "buy" else -1.0
        delta_gex += sign * gamma * qty * 100.0 * spot * spot * 0.01 / 1e9
    return {
        "flow_event_count": float(len(events)),
        "flow_net_delta_gex_bn": float(delta_gex),
        "flow_buy_ratio": float(buys / len(events)) if events else 0.0,
        "flow_aggressiveness": float(total_qty / len(events)) if events else 0.0,
    }


def merge_extended_features(
    metrics: dict[str, Any],
    *,
    greek_df: pd.DataFrame | None = None,
    spot_exposures_df: pd.DataFrame | None = None,
    market_date: str | None = None,
    vol_regime: dict[str, float] | None = None,
    cross_asset: dict[str, float] | None = None,
    flow_path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge all extended scalar features into a snapshot metrics dict."""
    spot = _safe_float(metrics.get("spot"), 0.0)
    merged: dict[str, float] = {}
    merged.update(summarize_greek_exposure_df(greek_df, spot) if greek_df is not None else {})
    merged.update(summarize_spot_exposures_df(spot_exposures_df) if spot_exposures_df is not None else {})
    merged.update(summarize_flow_feed(flow_path))
    merged.update(event_calendar_features(market_date or metrics.get("ts")))
    if vol_regime:
        merged.update(vol_regime)
    if cross_asset:
        merged.update(cross_asset)
    if "extended_features" in metrics and isinstance(metrics["extended_features"], dict):
        merged.update({k: _safe_float(v) for k, v in metrics["extended_features"].items()})
    metrics.update(merged)
    return apply_extended_defaults(metrics)
