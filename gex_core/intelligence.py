"""Higher-level intelligence helpers for dashboard and API layers."""

from __future__ import annotations

import ipaddress
import math
import os
import socket
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from gex_core.alert_config import load_alert_config
from gex_core.calibration import (
    expected_directional_move_pct as _fitted_move_pct,
    fit_close_above_flip_rate,
)
from gex_core.features import safe_float
from gex_core.history import build_history
from gex_core.models_manifest import load_manifest
from gex_core.predict import MIN_OVERLAY_TRAIN_ROWS

# Spot-scenario sensitivities. These remain heuristics, but are now named and
# documented rather than buried magic numbers. ``LOCAL_GEX_SENSITIVITY`` is the
# fraction of the local strike-GEX change that propagates to total GEX under a
# spot shift; ``FLIP_SHIFT_SENSITIVITY`` is how far the flip migrates per unit
# spot move.
LOCAL_GEX_SENSITIVITY = 0.65
FLIP_SHIFT_SENSITIVITY = 0.35
# Geometry vs empirical-base-rate blend for the close-above-flip probability.
FLIP_GEOMETRY_WEIGHT = 0.65


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _to_utc(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _selected_index(history: list[dict], selected_ts: str | None) -> int:
    if not history:
        return -1
    if not selected_ts:
        return len(history) - 1
    for idx, row in enumerate(history):
        if row.get("ts") == selected_ts:
            return idx
    return len(history) - 1


def build_today_regime_snapshot(selected: dict, prediction: dict | None) -> dict:
    """Build compact regime summary card payload."""
    confidence = safe_float(prediction.get("confidence"), 0.0) if prediction else 0.0
    spot = safe_float(selected.get("spot"), 0.0)
    gamma_flip = safe_float(selected.get("gamma_flip"), 0.0)
    flip_distance = abs(spot - gamma_flip) if spot > 0 and gamma_flip > 0 else None
    return {
        "regime": selected.get("regime", "N/A"),
        "total_gex": safe_float(selected.get("total_gex"), 0.0),
        "call_wall": selected.get("call_wall"),
        "put_wall": selected.get("put_wall"),
        "gamma_flip": selected.get("gamma_flip"),
        "spot": selected.get("spot"),
        "flip_distance_pts": flip_distance,
        "forecast_confidence": confidence,
        "predicted_regime": prediction.get("predicted_regime") if prediction else None,
    }


def _distance_payload(spot: float, level: float) -> dict | None:
    if spot <= 0 or level <= 0:
        return None
    pts = level - spot
    return {
        "level": level,
        "distance_pts": pts,
        "abs_distance_pts": abs(pts),
        "distance_pct": pts / spot,
        "abs_distance_pct": abs(pts) / spot,
        "side": "above spot" if pts > 0 else "below spot" if pts < 0 else "at spot",
    }


def build_gamma_analysis_panel(selected: dict, prediction: dict | None = None) -> dict:
    """Build an SPX-focused gamma analysis summary for dashboard cards."""
    spot = safe_float(selected.get("spot"), 0.0)
    total_gex = safe_float(selected.get("total_gex"), 0.0)
    pos_gex = safe_float(selected.get("pos_gex"), 0.0)
    neg_gex = safe_float(selected.get("neg_gex"), 0.0)
    gross_gex = abs(pos_gex) + abs(neg_gex)
    net_ratio = total_gex / gross_gex if gross_gex else 0.0
    gamma_flip = safe_float(selected.get("gamma_flip"), 0.0)
    call_wall = safe_float(selected.get("call_wall"), 0.0)
    put_wall = safe_float(selected.get("put_wall"), 0.0)
    near_term_ratio = safe_float(selected.get("near_term_ratio"), 0.0)
    concentration = safe_float(selected.get("gex_concentration"), 0.0)
    slope = safe_float(selected.get("cum_slope_at_spot"), 0.0)

    flip = _distance_payload(spot, gamma_flip)
    call = _distance_payload(spot, call_wall)
    put = _distance_payload(spot, put_wall)
    wall_candidates = [("Call wall", call), ("Put wall", put)]
    wall_candidates = [(name, payload) for name, payload in wall_candidates if payload is not None]
    nearest_wall_name = None
    nearest_wall = None
    if wall_candidates:
        nearest_wall_name, nearest_wall = min(wall_candidates, key=lambda item: item[1]["abs_distance_pts"])

    regime = str(selected.get("regime") or "")
    if "SHORT" in regime.upper():
        hedging_tone = "Short gamma: dealer hedging can amplify intraday SPX moves."
    elif "LONG" in regime.upper():
        hedging_tone = "Long gamma: dealer hedging can dampen SPX moves near key strikes."
    else:
        hedging_tone = "Neutral gamma: watch flip and wall proximity for regime definition."

    risk_score = 0.0
    if "SHORT" in regime.upper():
        risk_score += 35.0
    elif "LONG" in regime.upper():
        risk_score += 10.0
    if flip:
        risk_score += max(0.0, 30.0 * (1.0 - min(flip["abs_distance_pct"], 0.025) / 0.025))
    if nearest_wall:
        risk_score += max(0.0, 20.0 * (1.0 - min(nearest_wall["abs_distance_pct"], 0.015) / 0.015))
    risk_score += min(15.0, abs(near_term_ratio) * 15.0)
    risk_score = min(100.0, risk_score)

    if risk_score >= 70:
        risk_label = "high"
    elif risk_score >= 40:
        risk_label = "moderate"
    else:
        risk_label = "low"

    predicted_delta = safe_float(prediction.get("predicted_delta_gex"), 0.0) if prediction else 0.0
    return {
        "spot": spot,
        "regime": selected.get("regime", "N/A"),
        "hedging_tone": hedging_tone,
        "total_gex": total_gex,
        "positive_gex": pos_gex,
        "negative_gex": neg_gex,
        "gross_gex": gross_gex,
        "net_ratio": net_ratio,
        "gamma_flip": gamma_flip or None,
        "flip": flip,
        "spot_minus_flip_pts": spot - gamma_flip if spot > 0 and gamma_flip > 0 else None,
        "spot_minus_flip_pct": (spot - gamma_flip) / spot if spot > 0 and gamma_flip > 0 else None,
        "call_wall": call_wall or None,
        "put_wall": put_wall or None,
        "call": call,
        "put": put,
        "nearest_wall_name": nearest_wall_name,
        "nearest_wall": nearest_wall,
        "wall_spread": call_wall - put_wall if call_wall and put_wall else None,
        "near_term_ratio": near_term_ratio,
        "concentration": concentration,
        "cumulative_slope_at_spot": slope,
        "risk_score": risk_score,
        "risk_label": risk_label,
        "predicted_delta_gex": predicted_delta,
    }


def build_term_structure_panel(selected: dict, prediction: dict | None = None) -> dict:
    """Summarize 0DTE, near-term, and back-term GEX concentration."""
    current_zero = safe_float(selected.get("zero_dte_gex_bn"), 0.0)
    current_zero_ratio = safe_float(selected.get("zero_dte_ratio"), 0.0)
    near = safe_float(selected.get("near_term_gex_bn"), 0.0)
    near_ratio = safe_float(selected.get("near_term_ratio"), 0.0)
    back = safe_float(selected.get("back_term_gex_bn"), 0.0)
    back_ratio = safe_float(selected.get("back_term_ratio"), 0.0)
    curvature = safe_float(selected.get("term_curvature"), 0.0)

    predicted_zero_ratio = (
        safe_float(prediction.get("predicted_zero_dte_ratio"), current_zero_ratio)
        if prediction
        else current_zero_ratio
    )
    predicted_near_ratio = (
        safe_float(prediction.get("predicted_near_term_ratio"), near_ratio)
        if prediction
        else near_ratio
    )
    predicted_curvature = (
        safe_float(prediction.get("predicted_term_curvature"), curvature)
        if prediction
        else curvature
    )

    if abs(current_zero_ratio) >= 0.45:
        concentration_label = "0DTE-heavy"
    elif abs(near_ratio) >= 0.55:
        concentration_label = "near-term-heavy"
    elif abs(back_ratio) >= 0.55:
        concentration_label = "back-term-heavy"
    else:
        concentration_label = "balanced"

    return {
        "zero_dte_gex": current_zero,
        "zero_dte_ratio": current_zero_ratio,
        "near_term_gex": near,
        "near_term_ratio": near_ratio,
        "back_term_gex": back,
        "back_term_ratio": back_ratio,
        "term_curvature": curvature,
        "predicted_zero_dte_ratio": predicted_zero_ratio,
        "predicted_near_term_ratio": predicted_near_ratio,
        "predicted_term_curvature": predicted_curvature,
        "zero_dte_ratio_delta_forecast": predicted_zero_ratio - current_zero_ratio,
        "near_term_ratio_delta_forecast": predicted_near_ratio - near_ratio,
        "term_curvature_delta_forecast": predicted_curvature - curvature,
        "expiration_count": int(safe_float(selected.get("expiration_count"), 0.0)),
        "concentration_label": concentration_label,
    }


def build_model_accountability_panel(
    ticker: str,
    prediction: dict | None,
    backtest: dict | None = None,
) -> dict:
    """Expose model provenance, sample depth, and validation warnings."""
    manifest = load_manifest(ticker)
    backtest = backtest or {}
    training_count = int(safe_float(prediction.get("training_snapshot_count"), 0.0)) if prediction else 0
    confidence = safe_float(prediction.get("confidence"), 0.0) if prediction else 0.0
    raw_confidence = safe_float(prediction.get("raw_confidence"), confidence) if prediction else 0.0

    warnings: list[str] = []
    if training_count and training_count < 8:
        warnings.append("Recent training window is thin; confidence is sample-damped.")
    elif not training_count:
        warnings.append("No forecast was generated for this snapshot.")
    if backtest.get("n", 0) and backtest.get("accuracy") is not None:
        gap = safe_float(backtest.get("confidence_accuracy_gap"), 0.0)
        if gap > 0.2:
            warnings.append("Recent confidence and realized accuracy are materially different.")
    elif backtest:
        warnings.append("Walk-forward validation has too few samples.")
    n_manifest_train = safe_float((manifest.get("metrics") or {}).get("n_train"), 0.0) if manifest else 0.0
    overlay_active = n_manifest_train >= MIN_OVERLAY_TRAIN_ROWS
    if manifest and not overlay_active:
        warnings.append(
            f"Trained overlay inactive ({int(n_manifest_train)} rows; need {MIN_OVERLAY_TRAIN_ROWS}+). KNN remains primary."
        )

    backtest_acc = backtest.get("accuracy")
    return {
        "model_type": manifest.get("model_type") if manifest else None,
        "overlay_active": overlay_active,
        "overlay_min_train_rows": MIN_OVERLAY_TRAIN_ROWS,
        "manifest_train_rows": int(n_manifest_train) if manifest else 0,
        "trained_at_utc": manifest.get("trained_at_utc") if manifest else None,
        "training_start_ts": manifest.get("training_start_ts") if manifest else None,
        "training_end_ts": manifest.get("training_end_ts") if manifest else None,
        "manifest_metrics": manifest.get("metrics", {}) if manifest else {},
        "training_snapshot_count": training_count,
        "training_window_days": prediction.get("training_window_days") if prediction else None,
        "confidence": confidence,
        "raw_confidence": raw_confidence,
        "confidence_breakdown": prediction.get("confidence_breakdown", {}) if prediction else {},
        "backtest": backtest,
        "backtest_sign_accuracy": backtest_acc,
        "warnings": warnings,
    }


def generate_alerts(
    history: list[dict],
    selected: dict,
    prediction: dict | None = None,
    wall_shift_threshold: float | None = None,
) -> list[dict]:
    """Compute rule-based alerts for regime and structure changes."""
    cfg = load_alert_config()
    wall_shift_threshold = (
        wall_shift_threshold if wall_shift_threshold is not None else cfg.wall_shift_threshold
    )
    alerts: list[dict] = []
    idx = _selected_index(history, selected.get("ts"))
    prev = history[idx - 1] if idx > 0 else None

    if prev:
        if prev.get("regime") != selected.get("regime"):
            alerts.append(
                {
                    "severity": "high",
                    "title": "Regime shift detected",
                    "detail": f"{prev.get('regime', 'N/A')} → {selected.get('regime', 'N/A')}",
                }
            )

        for label, field in (("Call wall", "call_wall"), ("Put wall", "put_wall")):
            cur = safe_float(selected.get(field), 0.0)
            prv = safe_float(prev.get(field), 0.0)
            if cur and prv and abs(cur - prv) >= wall_shift_threshold:
                alerts.append(
                    {
                        "severity": "medium",
                        "title": f"{label} migration",
                        "detail": f"{prv:.0f} → {cur:.0f} ({cur - prv:+.0f} pts)",
                    }
                )

        prev_spot = safe_float(prev.get("spot"), 0.0)
        prev_flip = safe_float(prev.get("gamma_flip"), 0.0)
        cur_spot = safe_float(selected.get("spot"), 0.0)
        cur_flip = safe_float(selected.get("gamma_flip"), 0.0)
        if prev_spot and prev_flip and cur_spot and cur_flip:
            prev_side = prev_spot - prev_flip
            cur_side = cur_spot - cur_flip
            if prev_side == 0 or cur_side == 0 or (prev_side > 0) != (cur_side > 0):
                alerts.append(
                    {
                        "severity": "high",
                        "title": "Gamma flip crossing",
                        "detail": "Spot moved across the estimated flip zone.",
                    }
                )

        near_ratio_prev = safe_float(prev.get("near_term_ratio"), 0.0)
        near_ratio_cur = safe_float(selected.get("near_term_ratio"), 0.0)
        if (near_ratio_cur - near_ratio_prev) >= cfg.near_term_spike_threshold:
            alerts.append(
                {
                    "severity": "medium",
                    "title": "0DTE / near-term concentration spike",
                    "detail": f"Near-term ratio increased by {(near_ratio_cur - near_ratio_prev) * 100:.1f} pts.",
                }
            )

    if prediction:
        flip_prob = safe_float(prediction.get("regime_flip_probability"), 0.0)
        confidence = safe_float(prediction.get("confidence"), 0.0)
        delta = safe_float(prediction.get("predicted_delta_gex"), 0.0)
        total = abs(safe_float(selected.get("total_gex"), 0.0))
        if flip_prob >= cfg.regime_flip_prob_threshold:
            alerts.append(
                {
                    "severity": "high" if flip_prob >= 0.7 else "medium",
                    "title": "Elevated regime-flip probability",
                    "detail": f"Model estimates {flip_prob * 100:.1f}% chance of regime transition.",
                }
            )
        if abs(delta) > max(cfg.large_delta_gex_bn, total * cfg.large_delta_gex_ratio):
            alerts.append(
                {
                    "severity": "medium",
                    "title": "Large forecasted ΔGEX",
                    "detail": f"Projected next change: {delta:+.3f} Bn$ / %. (confidence {confidence * 100:.0f}%)",
                }
            )

    severity_rank = {"high": 3, "medium": 2, "low": 1}
    alerts.sort(key=lambda row: severity_rank.get(row["severity"], 0), reverse=True)
    return alerts


def compute_forecast_probabilities(
    selected: dict,
    prediction: dict | None,
    history: list[dict],
) -> dict | None:
    """Estimate probabilistic outcomes around flip and expected move."""
    if not prediction:
        return None

    spot = safe_float(selected.get("spot"), 0.0)
    pred_flip = safe_float(prediction.get("predicted_flip"), safe_float(selected.get("gamma_flip"), 0.0))
    confidence = safe_float(prediction.get("confidence"), 0.0)
    flip_prob = safe_float(prediction.get("regime_flip_probability"), 0.0)
    distance = 0.0 if not spot else (spot - pred_flip) / max(abs(spot), 1.0)

    # Start from spot-vs-flip geometry and blend with model certainty, then
    # anchor on the empirical close-above-flip base rate when we have enough
    # history to estimate it (replaces the un-calibrated fixed blend).
    geometry = _sigmoid(distance * 24.0)
    base_rate = fit_close_above_flip_rate(history)
    if base_rate is not None:
        above_flip = FLIP_GEOMETRY_WEIGHT * geometry + (1.0 - FLIP_GEOMETRY_WEIGHT) * base_rate
    else:
        above_flip = (0.65 * geometry) + (0.35 * (1.0 - (flip_prob * 0.5)))
    above_flip = min(0.99, max(0.01, above_flip))
    below_flip = 1.0 - above_flip

    spot_moves: list[float] = []
    for idx in range(len(history) - 1):
        s0 = safe_float(history[idx].get("spot"), 0.0)
        s1 = safe_float(history[idx + 1].get("spot"), 0.0)
        if s0 > 0 and s1 > 0:
            spot_moves.append((s1 - s0) / s0)
    expected_abs_move_pct = float(sum(abs(x) for x in spot_moves) / len(spot_moves)) if spot_moves else 0.0
    # Fitted ΔGEX -> forward-return slope instead of the hard-coded 0.00035.
    expected_directional_move_pct = _fitted_move_pct(
        safe_float(prediction.get("predicted_delta_gex"), 0.0), history
    )

    return {
        "prob_close_above_flip": above_flip,
        "prob_close_below_flip": below_flip,
        "prob_regime_flip": flip_prob,
        "expected_abs_move_pct": expected_abs_move_pct,
        "expected_directional_move_pct": expected_directional_move_pct,
        "flip_base_rate": base_rate,
        "confidence": confidence,
    }


def compute_confluence_overlay(
    selected: dict,
    prediction: dict | None,
    flow_overlay: dict | None,
) -> dict:
    """Compute confluence score combining structure, model, and flow."""
    spot = safe_float(selected.get("spot"), 0.0)
    gamma_flip = safe_float(selected.get("gamma_flip"), 0.0)
    call_wall = safe_float(selected.get("call_wall"), 0.0)
    put_wall = safe_float(selected.get("put_wall"), 0.0)
    total = abs(safe_float(selected.get("total_gex"), 0.0))

    flip_component = 0.0
    if spot > 0 and gamma_flip > 0:
        flip_dist_pct = abs(spot - gamma_flip) / spot
        flip_component = max(0.0, 30.0 * (1.0 - (flip_dist_pct / 0.03)))

    wall_component = 0.0
    if spot > 0 and call_wall and put_wall:
        nearest_wall = min(abs(spot - call_wall), abs(spot - put_wall)) / spot
        wall_component = max(0.0, 20.0 * (1.0 - (nearest_wall / 0.03)))

    model_component = min(20.0, safe_float(prediction.get("confidence"), 0.0) * 20.0) if prediction else 0.0

    flow_component = 0.0
    if flow_overlay and flow_overlay.get("event_count", 0) > 0:
        events = int(flow_overlay["event_count"])
        flow_component += min(12.0, math.log1p(events) * 3.0)
        pred_delta = safe_float(prediction.get("predicted_delta_gex"), 0.0) if prediction else 0.0
        flow_delta = safe_float(flow_overlay.get("predicted_flow_delta_gex_bn"), 0.0)
        if pred_delta == 0 or flow_delta == 0:
            flow_component += 4.0
        elif (pred_delta > 0) == (flow_delta > 0):
            flow_component += 8.0
        else:
            flow_component += 2.0

    stability_component = min(10.0, total / 8.0)
    score = min(100.0, flip_component + wall_component + model_component + flow_component + stability_component)
    if score >= 75:
        label = "high"
    elif score >= 45:
        label = "medium"
    else:
        label = "low"

    return {
        "score": score,
        "label": label,
        "components": [
            {"name": "Flip proximity", "score": flip_component, "max": 30},
            {"name": "Wall proximity", "score": wall_component, "max": 20},
            {"name": "Forecast confidence", "score": model_component, "max": 20},
            {"name": "Flow alignment", "score": flow_component, "max": 20},
            {"name": "Regime stability", "score": stability_component, "max": 10},
        ],
    }


def simulate_spot_scenario(selected: dict, spot_shift_pct: float) -> dict | None:
    """What-if simulation for spot shift impact on structure."""
    spot = safe_float(selected.get("spot"), 0.0)
    if spot <= 0:
        return None
    strike = selected.get("strike")
    if strike is None:
        return None

    strike_series = pd.Series(strike, dtype=float).sort_index()
    if strike_series.empty:
        return None

    new_spot = spot * (1.0 + spot_shift_pct)
    total = safe_float(selected.get("total_gex"), 0.0)
    idx = strike_series.index.astype(float).to_numpy()
    vals = strike_series.to_numpy(dtype=float)
    local_now = float(np.interp(spot, idx, vals))
    local_new = float(np.interp(new_spot, idx, vals))
    projected_total = total + ((local_new - local_now) * LOCAL_GEX_SENSITIVITY)

    gamma_flip = selected.get("gamma_flip")
    projected_flip = None
    if gamma_flip is not None:
        projected_flip = safe_float(gamma_flip) - (spot_shift_pct * spot * FLIP_SHIFT_SENSITIVITY)

    window = max(spot * 0.02, 1.0)
    local_band = strike_series.loc[(strike_series.index >= new_spot - window) & (strike_series.index <= new_spot + window)]
    if len(local_band) < 3:
        local_band = strike_series

    projected_call = float(local_band.idxmax()) if len(local_band) else None
    projected_put = float(local_band.idxmin()) if len(local_band) else None

    return {
        "spot_shift_pct": spot_shift_pct,
        "new_spot": new_spot,
        "projected_total_gex": projected_total,
        "projected_regime": "LONG gamma" if projected_total >= 0 else "SHORT gamma",
        "projected_flip": projected_flip,
        "projected_call_wall": projected_call,
        "projected_put_wall": projected_put,
    }


def build_strategy_assistant(selected: dict, prediction: dict | None, confluence: dict | None) -> list[str]:
    """Educational strategy guidance tied to regime state."""
    regime = str(selected.get("regime", "N/A")).upper()
    confidence = safe_float(prediction.get("confidence"), 0.0) if prediction else 0.0
    confluence_label = confluence.get("label", "low") if confluence else "low"
    notes: list[str] = []

    if "LONG" in regime:
        notes.append("Long-gamma conditions often favor mean-reversion around major walls.")
        notes.append("Watch for failed breakouts near call wall and support bounces near put wall.")
    elif "SHORT" in regime:
        notes.append("Short-gamma conditions often amplify momentum once key walls break.")
        notes.append("Trend continuation risk increases when spot is far from gamma flip.")
    else:
        notes.append("Neutral gamma can produce mixed behavior; prioritize confirmation from flow and breadth.")

    if prediction:
        notes.append(
            f"Model confidence is {confidence * 100:.0f}% with projected regime {prediction.get('predicted_regime', 'N/A')}."
        )
    if confluence_label == "high":
        notes.append("High confluence suggests structure + flow are aligned; expect cleaner reactions at key levels.")
    elif confluence_label == "medium":
        notes.append("Medium confluence suggests selective setups; avoid forcing trades between major levels.")
    else:
        notes.append("Low confluence suggests noisy structure; reduce size and require stronger confirmation.")
    return notes


def build_data_quality_panel(selected: dict, history: list[dict]) -> dict:
    """Compute data-trust diagnostics for current snapshot."""
    strike = pd.Series(selected.get("strike"), dtype=float) if selected.get("strike") is not None else pd.Series(dtype=float)
    contracts = int(len(strike))
    non_zero_ratio = float((strike != 0).mean()) if contracts else 0.0
    top5_concentration = float(strike.abs().sort_values(ascending=False).head(5).sum() / max(strike.abs().sum(), 1e-9)) if contracts else 0.0

    gamma_flip = selected.get("gamma_flip")
    flip_in_range = True
    if contracts and gamma_flip is not None:
        low = float(strike.index.min())
        high = float(strike.index.max())
        flip_val = safe_float(gamma_flip)
        flip_in_range = low <= flip_val <= high

    age_minutes = None
    ts_value = _to_utc(selected.get("ts"))
    if ts_value is not None:
        age_minutes = (datetime.now(timezone.utc) - ts_value).total_seconds() / 60.0

    trust_score = 100.0
    if contracts < 25:
        trust_score -= 22
    if non_zero_ratio < 0.7:
        trust_score -= 12
    if top5_concentration > 0.7:
        trust_score -= 12
    if not flip_in_range:
        trust_score -= 18
    if age_minutes is not None and age_minutes > 120:
        trust_score -= 18
    if len(history) < 6:
        trust_score -= 8
    trust_score = max(0.0, min(100.0, trust_score))

    flags: list[str] = []
    if contracts < 25:
        flags.append("Limited strike depth in current export.")
    if top5_concentration > 0.7:
        flags.append("Exposure heavily concentrated in a few strikes.")
    if not flip_in_range:
        flags.append("Estimated gamma flip lies outside observed strike range.")
    if age_minutes is not None and age_minutes > 120:
        flags.append("Snapshot is stale relative to intraday cadence.")
    if not flags:
        flags.append("No major quality warnings detected.")

    return {
        "trust_score": trust_score,
        "contracts": contracts,
        "non_zero_ratio": non_zero_ratio,
        "top5_concentration": top5_concentration,
        "flip_in_range": flip_in_range,
        "age_minutes": age_minutes,
        "history_depth": len(history),
        "flags": flags,
    }


def build_outcome_panel(history: list[dict], selected_ts: str | None) -> dict | None:
    """Compute forward outcomes and by-regime realized behavior."""
    if len(history) < 2:
        return None

    idx = _selected_index(history, selected_ts)
    selected = history[idx]
    selected_spot = safe_float(selected.get("spot"), 0.0)

    horizon_rows = []
    for horizon in (1, 3):
        target_idx = idx + horizon
        if selected_spot <= 0 or target_idx >= len(history):
            continue
        target_spot = safe_float(history[target_idx].get("spot"), 0.0)
        if target_spot <= 0:
            continue
        pts = target_spot - selected_spot
        pct = pts / selected_spot
        horizon_rows.append(
            {
                "horizon": horizon,
                "spot": target_spot,
                "move_points": pts,
                "move_pct": pct,
            }
        )

    regime_buckets: dict[str, list[float]] = {}
    for i in range(len(history) - 1):
        s0 = safe_float(history[i].get("spot"), 0.0)
        s1 = safe_float(history[i + 1].get("spot"), 0.0)
        if s0 <= 0 or s1 <= 0:
            continue
        regime = history[i].get("regime", "N/A")
        regime_buckets.setdefault(regime, []).append((s1 - s0) / s0)

    regime_stats = []
    for regime, values in regime_buckets.items():
        if not values:
            continue
        positives = sum(1 for v in values if v > 0)
        abs_values = [abs(v) for v in values]
        regime_stats.append(
            {
                "regime": regime,
                "samples": len(values),
                "avg_move_pct": sum(values) / len(values),
                "avg_abs_move_pct": sum(abs_values) / len(abs_values),
                "hit_rate_up": positives / len(values),
                "hit_rate_down": 1.0 - (positives / len(values)),
            }
        )

    flip_crosses = 0
    wall_interactions = 0
    checked = 0
    for i in range(len(history) - 1):
        spot0 = safe_float(history[i].get("spot"), 0.0)
        spot1 = safe_float(history[i + 1].get("spot"), 0.0)
        flip = safe_float(history[i].get("gamma_flip"), 0.0)
        call_wall = safe_float(history[i].get("call_wall"), 0.0)
        put_wall = safe_float(history[i].get("put_wall"), 0.0)
        if spot0 <= 0 or spot1 <= 0:
            continue
        checked += 1
        if flip > 0 and (spot0 - flip == 0 or spot1 - flip == 0 or (spot0 > flip) != (spot1 > flip)):
            flip_crosses += 1
        if call_wall > 0 and min(spot0, spot1) <= call_wall <= max(spot0, spot1):
            wall_interactions += 1
        elif put_wall > 0 and min(spot0, spot1) <= put_wall <= max(spot0, spot1):
            wall_interactions += 1

    return {
        "horizons": horizon_rows,
        "regime_stats": regime_stats,
        "structure_stats": {
            "samples": checked,
            "flip_cross_rate": flip_crosses / checked if checked else None,
            "wall_interaction_rate": wall_interactions / checked if checked else None,
        },
        "selected_ts": selected.get("ts_label"),
    }


def build_watchlist_rows(tickers: list[str]) -> list[dict]:
    """Aggregate supported dashboard ticker metrics."""
    rows = []
    for ticker in tickers:
        history = build_history(ticker)
        if not history:
            continue
        latest = history[-1]
        confluence = compute_confluence_overlay(latest, prediction=None, flow_overlay=None)
        total = safe_float(latest.get("total_gex"), 0.0)
        spot = safe_float(latest.get("spot"), 0.0)
        gamma_flip = safe_float(latest.get("gamma_flip"), 0.0)
        call_wall = safe_float(latest.get("call_wall"), 0.0)
        put_wall = safe_float(latest.get("put_wall"), 0.0)
        flip_distance_pct = abs(spot - gamma_flip) / spot if spot > 0 and gamma_flip > 0 else None
        nearest_wall = min(abs(spot - call_wall), abs(spot - put_wall)) if spot > 0 and call_wall and put_wall else None
        wall_proximity_pct = (nearest_wall / spot) if nearest_wall is not None and spot > 0 else None

        lookback = history[-6:]
        if len(lookback) > 1:
            totals = [safe_float(x.get("total_gex"), 0.0) for x in lookback]
            mean_abs = sum(abs(x) for x in totals) / len(totals)
            variance = sum((x - (sum(totals) / len(totals))) ** 2 for x in totals) / len(totals)
            std_dev = math.sqrt(variance)
            stability = 1.0 - min(1.0, std_dev / max(mean_abs, 1.0))
        else:
            stability = 0.5

        rows.append(
            {
                "ticker": ticker,
                "history_count": len(history),
                "latest_ts": latest.get("ts_label", "N/A"),
                "total_gex": total,
                "regime": latest.get("regime", "N/A"),
                "flip_distance_pct": flip_distance_pct,
                "wall_proximity_pct": wall_proximity_pct,
                "regime_stability": stability,
                "confluence_score": confluence["score"],
            }
        )

    return sorted(rows, key=lambda row: abs(row.get("total_gex", 0.0)), reverse=True)


def validate_webhook_url(url: str) -> tuple[bool, str]:
    """Validate a webhook URL against SSRF abuse.

    Requires HTTPS and rejects hosts that resolve to private, loopback,
    link-local, or otherwise non-public addresses (e.g. cloud metadata
    endpoints like ``169.254.169.254``). Set ``GEX_ALERT_ALLOW_INSECURE=1``
    to permit ``http`` and private targets for local testing.
    """
    allow_insecure = os.environ.get("GEX_ALERT_ALLOW_INSECURE", "").lower() in {"1", "true", "yes"}
    parsed = urlparse(url)
    if parsed.scheme not in {"https"} and not (allow_insecure and parsed.scheme == "http"):
        return False, "Webhook URL must use https."
    host = parsed.hostname
    if not host:
        return False, "Webhook URL has no host."
    if allow_insecure:
        return True, "ok"
    try:
        addrinfos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        return False, f"Webhook host did not resolve: {exc}"
    for info in addrinfos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False, f"Webhook host resolves to non-public address {ip}."
    return True, "ok"


def dispatch_alerts_to_webhook(ticker: str, alerts: list[dict]) -> tuple[bool, str]:
    """Dispatch alerts to webhook if configured."""
    url = os.environ.get("GEX_ALERT_WEBHOOK_URL")
    if not url:
        return False, "GEX_ALERT_WEBHOOK_URL not configured."
    if not alerts:
        return False, "No alerts to dispatch."
    ok, reason = validate_webhook_url(url)
    if not ok:
        return False, f"Webhook URL rejected: {reason}"
    payload = {
        "ticker": ticker,
        "alert_count": len(alerts),
        "alerts": alerts,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        resp = requests.post(url, json=payload, timeout=7)
        if 200 <= resp.status_code < 300:
            return True, f"Dispatched {len(alerts)} alerts."
        return False, f"Webhook returned status {resp.status_code}."
    except requests.RequestException as exc:
        return False, f"Webhook error: {exc}"
