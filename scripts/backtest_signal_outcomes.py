"""Backtest whether GEX-derived signals predict forward spot behavior."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.exports import EXPORT_DIR
from gex_core.features import safe_float
from gex_core.history import build_history
from gex_core.intelligence import compute_confluence_overlay, compute_forecast_probabilities
from gex_core.predict import predict_next_snapshot


def _wall_respected(spot0: float, spot1: float, wall: float | None, *, side: str) -> bool | None:
    if wall is None or spot0 <= 0 or spot1 <= 0:
        return None
    if side == "call":
        return spot1 <= wall if spot0 < wall else spot1 > wall
    return spot1 >= wall if spot0 > wall else spot1 < wall


def backtest_signals(ticker: str) -> dict:
    history = build_history(ticker, EXPORT_DIR)
    if len(history) < 4:
        return {"ticker": ticker, "samples": 0, "message": "Not enough history"}

    rows = []
    for i in range(len(history) - 1):
        cur = history[i]
        nxt = history[i + 1]
        pred = predict_next_snapshot(history[: i + 1])
        if not pred:
            continue
        spot0 = safe_float(cur.get("spot"), 0.0)
        spot1 = safe_float(nxt.get("spot"), 0.0)
        if spot0 <= 0 or spot1 <= 0:
            continue
        ret = (spot1 - spot0) / spot0
        flip = safe_float(cur.get("gamma_flip"), 0.0)
        probs = compute_forecast_probabilities(cur, pred)
        confluence = compute_confluence_overlay(cur, pred, flow_overlay=None)
        rows.append(
            {
                "flip_cross_correct": (spot1 > flip) == (spot0 > flip) if flip > 0 else None,
                "close_above_flip": spot1 > flip if flip > 0 else None,
                "call_wall_respected": _wall_respected(spot0, spot1, cur.get("call_wall"), side="call"),
                "put_wall_respected": _wall_respected(spot0, spot1, cur.get("put_wall"), side="put"),
                "direction_sign": 1.0 if ret > 0 else -1.0 if ret < 0 else 0.0,
                "predicted_delta_sign": 1.0 if safe_float(pred.get("predicted_delta_gex")) >= 0 else -1.0,
                "confluence": safe_float(confluence.get("score")),
                "prob_above_flip": safe_float(probs.get("close_above_flip")),
                "return_pct": ret,
            }
        )

    if not rows:
        return {"ticker": ticker, "samples": 0, "message": "No valid walk-forward rows"}

    def rate(key: str) -> float | None:
        vals = [row[key] for row in rows if row.get(key) is not None]
        if not vals:
            return None
        return float(sum(bool(v) for v in vals) / len(vals))

    high_conf = [r for r in rows if r["confluence"] >= 60]
    return {
        "ticker": ticker,
        "samples": len(rows),
        "flip_cross_accuracy": rate("flip_cross_correct"),
        "close_above_flip_rate": rate("close_above_flip"),
        "call_wall_respect_rate": rate("call_wall_respected"),
        "put_wall_respect_rate": rate("put_wall_respected"),
        "delta_sign_accuracy": (
            sum(
                1
                for r in rows
                if r["direction_sign"] != 0
                and (r["direction_sign"] > 0) == (r["predicted_delta_sign"] > 0)
            )
            / max(1, sum(1 for r in rows if r["direction_sign"] != 0))
        ),
        "high_confluence_samples": len(high_conf),
        "avg_return_pct": sum(r["return_pct"] for r in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest GEX signal outcomes")
    parser.add_argument("--ticker", default="SPX")
    args = parser.parse_args()
    result = backtest_signals(args.ticker.upper())
    print(f"\n=== Signal outcome backtest: {result['ticker']} ===")
    for key, value in result.items():
        if key == "ticker":
            continue
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
