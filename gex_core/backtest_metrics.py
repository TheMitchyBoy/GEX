"""Rolling backtest metrics for prediction confidence display."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from gex_core.history import build_history
from gex_core.predict import predict_next_snapshot


@lru_cache(maxsize=16)
def backtest_delta_sign_accuracy(ticker: str, min_history: int = 6) -> dict[str, Any]:
    """
    Walk-forward: predict next delta GEX sign vs actual from export history.
    Cached per ticker for dashboard display.
    """
    history = build_history(ticker.upper())
    if len(history) < min_history:
        return {"n": 0, "accuracy": None, "mae_delta": None}

    hits = 0
    total = 0
    abs_errors = []
    enriched = [enrich_snapshot_metrics(h.copy()) for h in history]

    for i in range(4, len(enriched) - 1):
        window = enriched[: i + 1]
        pred = predict_next_snapshot(window)
        if not pred:
            continue
        actual_delta = enriched[i + 1]["total_gex"] - enriched[i]["total_gex"]
        predicted_delta = pred["predicted_delta_gex"]
        if (actual_delta >= 0) == (predicted_delta >= 0):
            hits += 1
        total += 1
        abs_errors.append(abs(actual_delta - predicted_delta))

    return {
        "n": total,
        "accuracy": (hits / total) if total else None,
        "mae_delta": float(np.mean(abs_errors)) if abs_errors else None,
    }
