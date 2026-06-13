"""Rolling backtest metrics for prediction confidence display.

Disabled by default on the web app (``GEX_BACKTEST_METRICS=0``) so dashboard and
refresh paths never load walk-forward history. Enable explicitly for CI or
offline scripts.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from gex_core.calibration import calibrate_confidence

_EMPTY: dict[str, Any] = {
    "n": 0,
    "accuracy": None,
    "mae_delta": None,
    "avg_confidence": None,
    "confidence_accuracy_gap": None,
    "baseline_accuracy": None,
    "baseline_momentum_accuracy": None,
    "accuracy_by_regime": {},
    "regime_flip_recall": None,
    "regime_flip_events": 0,
}


def backtest_metrics_enabled() -> bool:
    return os.environ.get("GEX_BACKTEST_METRICS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _backtest_limits() -> tuple[int, int, int]:
    max_folds = int(os.environ.get("GEX_BACKTEST_MAX_FOLDS", "24"))
    lookback_days = int(os.environ.get("GEX_BACKTEST_LOOKBACK_DAYS", "30"))
    max_snapshots = int(os.environ.get("GEX_BACKTEST_MAX_SNAPSHOTS", "80"))
    return max_folds, lookback_days, max_snapshots


def backtest_delta_sign_accuracy(
    ticker: str,
    min_history: int = 6,
    *,
    history: list[dict] | None = None,
) -> dict[str, Any]:
    """Walk-forward: predict next ΔGEX sign vs actual from export history."""
    if not backtest_metrics_enabled():
        return dict(_EMPTY)

    from gex_core.history import build_history
    from gex_core.predict import predict_next_snapshot

    ticker = ticker.upper()
    max_folds, lookback_days, max_snapshots = _backtest_limits()
    if history is None:
        history = build_history(
            ticker,
            lookback_days=lookback_days,
            max_snapshots=max_snapshots,
        )
    elif lookback_days and lookback_days > 0 and history:
        from datetime import timedelta

        from gex_core.exports import parse_timestamp

        latest = parse_timestamp(history[-1]["ts"])
        since = latest - timedelta(days=lookback_days)
        history = [row for row in history if parse_timestamp(row["ts"]) >= since]
        if max_snapshots and len(history) > max_snapshots:
            history = history[-max_snapshots:]
    if len(history) < min_history:
        return dict(_EMPTY)

    hits = 0
    total = 0
    baseline_hits = 0
    baseline_total = 0
    abs_errors = []
    confidences = []
    regime_hits: dict[str, int] = {}
    regime_totals: dict[str, int] = {}
    flip_alerts = 0
    actual_flips = 0

    fold_indices = list(range(4, len(history) - 1))
    if len(fold_indices) > max_folds:
        step = max(1, len(fold_indices) // max_folds)
        fold_indices = fold_indices[::step][:max_folds]

    for i in fold_indices:
        window = history[: i + 1]
        pred = predict_next_snapshot(window)
        if not pred:
            continue
        actual_delta = history[i + 1]["total_gex"] - history[i]["total_gex"]
        predicted_delta = pred["predicted_delta_gex"]
        sign_hit = (actual_delta >= 0) == (predicted_delta >= 0)
        if sign_hit:
            hits += 1
        total += 1
        abs_errors.append(abs(actual_delta - predicted_delta))
        confidences.append(float(pred.get("confidence", 0.0)))

        regime = history[i].get("regime", "N/A")
        regime_totals[regime] = regime_totals.get(regime, 0) + 1
        regime_hits[regime] = regime_hits.get(regime, 0) + int(sign_hit)

        prev_delta = history[i]["total_gex"] - history[i - 1]["total_gex"]
        if prev_delta != 0 or actual_delta != 0:
            baseline_total += 1
            if prev_delta == 0:
                baseline_hit = abs(actual_delta) < 1e-9
            elif actual_delta == 0:
                baseline_hit = False
            else:
                baseline_hit = (prev_delta >= 0) == (actual_delta >= 0)
            baseline_hits += int(baseline_hit)

        flipped = (history[i]["total_gex"] >= 0) != (history[i + 1]["total_gex"] >= 0)
        if flipped:
            actual_flips += 1
            if float(pred.get("regime_flip_probability", 0.0)) >= 0.3:
                flip_alerts += 1

    accuracy = (hits / total) if total else None
    avg_confidence = float(np.mean(confidences)) if confidences else None

    return {
        "n": total,
        "accuracy": accuracy,
        "mae_delta": float(np.mean(abs_errors)) if abs_errors else None,
        "avg_confidence": avg_confidence,
        "confidence_accuracy_gap": (
            abs(avg_confidence - accuracy) if avg_confidence is not None and accuracy is not None else None
        ),
        "baseline_accuracy": (baseline_hits / baseline_total) if baseline_total else None,
        "baseline_momentum_accuracy": (baseline_hits / baseline_total) if baseline_total else None,
        "accuracy_by_regime": {
            regime: {
                "n": regime_totals[regime],
                "accuracy": regime_hits.get(regime, 0) / regime_totals[regime],
            }
            for regime in sorted(regime_totals)
        },
        "regime_flip_recall": (flip_alerts / actual_flips) if actual_flips else None,
        "regime_flip_events": actual_flips,
    }


def calibrated_prediction_confidence(ticker: str, raw_confidence: float) -> float:
    """Calibrate a raw forecast confidence against backtest + logged LLM outcomes."""
    if not backtest_metrics_enabled():
        return raw_confidence
    bt = backtest_delta_sign_accuracy(ticker)
    calibrated = calibrate_confidence(raw_confidence, bt.get("accuracy"), bt.get("n", 0) or 0)
    try:
        from gex_core.prediction_log import get_llm_calibration_stats

        llm_stats = get_llm_calibration_stats(ticker)
        if (llm_stats.get("n") or 0) >= int(os.environ.get("GEX_LLM_CALIB_MIN_SAMPLES", "5")):
            return calibrate_confidence(
                calibrated,
                llm_stats.get("sign_accuracy"),
                llm_stats.get("n", 0) or 0,
            )
    except Exception:
        pass
    return calibrated
