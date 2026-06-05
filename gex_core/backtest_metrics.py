"""Rolling backtest metrics for prediction confidence display.

The cache is keyed on a signature of the export set (latest timestamp + file
count) so it invalidates automatically when new snapshots land during the
server's lifetime -- the previous ``lru_cache(ticker)`` went stale.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gex_core.calibration import calibrate_confidence
from gex_core.history import build_history, collect_snapshot_files
from gex_core.predict import predict_next_snapshot

_CACHE: dict[tuple[str, str, int], dict[str, Any]] = {}


def _export_signature(ticker: str) -> tuple[str, int]:
    files = collect_snapshot_files(ticker.upper())
    if not files:
        return ("", 0)
    return (max(files.keys()), len(files))


def backtest_delta_sign_accuracy(ticker: str, min_history: int = 6) -> dict[str, Any]:
    """Walk-forward: predict next ΔGEX sign vs actual from export history.

    Cached per export signature for dashboard display. Also reports ΔGEX MAE and
    a naive momentum baseline so the dashboard can show skill *vs* the baseline.
    """
    ticker = ticker.upper()
    sig_ts, sig_n = _export_signature(ticker)
    cache_key = (ticker, sig_ts, sig_n)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    history = build_history(ticker)
    empty = {
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
    if len(history) < min_history:
        _CACHE[cache_key] = empty
        return empty

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

    for i in range(4, len(history) - 1):
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

    result = {
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
    _CACHE[cache_key] = result
    return result


def calibrated_prediction_confidence(ticker: str, raw_confidence: float) -> float:
    """Calibrate a raw forecast confidence against the rolling backtest hit-rate."""
    bt = backtest_delta_sign_accuracy(ticker)
    return calibrate_confidence(raw_confidence, bt.get("accuracy"), bt.get("n", 0) or 0)


def clear_cache() -> None:
    _CACHE.clear()
