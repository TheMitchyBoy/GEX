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
    empty = {"n": 0, "accuracy": None, "mae_delta": None, "baseline_accuracy": None}
    if len(history) < min_history:
        _CACHE[cache_key] = empty
        return empty

    hits = 0
    total = 0
    abs_errors = []
    baseline_hits = 0
    baseline_total = 0

    for i in range(4, len(history) - 1):
        window = history[: i + 1]
        pred = predict_next_snapshot(window)
        if not pred:
            continue
        actual_delta = history[i + 1]["total_gex"] - history[i]["total_gex"]
        predicted_delta = pred["predicted_delta_gex"]
        if (actual_delta >= 0) == (predicted_delta >= 0):
            hits += 1
        total += 1
        abs_errors.append(abs(actual_delta - predicted_delta))

        prev_delta = history[i]["total_gex"] - history[i - 1]["total_gex"]
        if (actual_delta >= 0) == (prev_delta >= 0):
            baseline_hits += 1
        baseline_total += 1

    result = {
        "n": total,
        "accuracy": (hits / total) if total else None,
        "mae_delta": float(np.mean(abs_errors)) if abs_errors else None,
        "baseline_accuracy": (baseline_hits / baseline_total) if baseline_total else None,
    }
    _CACHE[cache_key] = result
    return result


def calibrated_prediction_confidence(ticker: str, raw_confidence: float) -> float:
    """Calibrate a raw forecast confidence against the rolling backtest hit-rate."""
    bt = backtest_delta_sign_accuracy(ticker)
    return calibrate_confidence(raw_confidence, bt.get("accuracy"), bt.get("n", 0) or 0)


def clear_cache() -> None:
    _CACHE.clear()
