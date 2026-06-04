"""Backtest GEX change predictions against historical exports.

Evaluates:
- KNN predictor (delta GEX sign and magnitude)
- Naive momentum signal (sign of previous delta)
- Optional trained model if present

Usage:
    python scripts/backtest_gex_prediction.py --ticker SPX
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.exports import find_exports_for_ticker, parse_timestamp
from gex_core.features import compute_features_from_exports
from gex_core.predict import predict_next_snapshot, prepare_training_rows
from gex_core.features import enrich_snapshot_metrics


def build_history_from_exports(ticker: str) -> list[dict]:
    exports = find_exports_for_ticker(ticker)
    timestamps = sorted(
        ts for ts, kinds in exports.items() if "gex_by_strike" in kinds
    )
    history = []
    prev_feats = None
    for ts in timestamps:
        info = exports[ts]
        feats = compute_features_from_exports(info, prev_features=prev_feats)
        prev_feats = feats
        strike_path = info.get("gex_by_strike")
        cumulative_path = info.get("cumulative_gex")
        import pandas as pd
        from gex_core.exports import load_strike_series, load_cumulative_series

        strike = load_strike_series(strike_path) if strike_path else pd.Series(dtype=float)
        cumulative = load_cumulative_series(cumulative_path) if cumulative_path else pd.Series(dtype=float)

        row = {
            "ts": ts,
            "ts_label": parse_timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "strike": strike,
            "cumulative": cumulative,
            "total_gex": feats["total_gex_bn"],
            "pos_gex": feats["pos_gex_bn"],
            "neg_gex": feats["neg_gex_bn"],
            "gex_std": feats["gex_std_bn"],
            "near_term_ratio": feats["near_term_ratio"],
            "surface_peak": feats.get("surface_peak", 0.0),
            "call_wall": feats.get("call_wall"),
            "put_wall": feats.get("put_wall"),
            "gamma_flip": feats.get("gamma_flip"),
            "regime": "LONG gamma" if feats["total_gex_bn"] >= 0 else "SHORT gamma",
            "abs_mean": abs(strike).mean() if len(strike) else 0.0,
        }
        history.append(enrich_snapshot_metrics(row))
    return history


def backtest(ticker: str):
    history = build_history_from_exports(ticker)
    if len(history) < 5:
        print(f"Need at least 5 snapshots; found {len(history)}")
        return

    rows = []
    for i in range(4, len(history) - 1):
        subset = history[: i + 1]
        actual_next = history[i + 1]
        current = subset[-1]
        actual_delta = actual_next["total_gex"] - current["total_gex"]

        pred = predict_next_snapshot(subset)
        if pred is None:
            continue

        pred_delta = pred["predicted_delta_gex"]
        pred_total = pred["predicted_total_gex"]

        rows.append(
            {
                "ts": current["ts_label"],
                "actual_delta": actual_delta,
                "predicted_delta": pred_delta,
                "actual_total": actual_next["total_gex"],
                "predicted_total": pred_total,
                "delta_sign_correct": int(np.sign(pred_delta) == np.sign(actual_delta)) if actual_delta != 0 else int(abs(pred_delta) < 1),
                "delta_mae": abs(actual_delta - pred_delta),
                "confidence": pred["confidence"],
            }
        )

    if not rows:
        print("Not enough data for walk-forward backtest.")
        return

    df = pd.DataFrame(rows)
    sign_acc = df["delta_sign_correct"].mean()
    mae = df["delta_mae"].mean()

    print(f"\n=== GEX Δ Prediction Backtest: {ticker} ===")
    print(f"Samples:              {len(df)}")
    print(f"ΔGEX sign accuracy:   {sign_acc:.3f}")
    print(f"ΔGEX MAE:             {mae:.4f} Bn$ / %")
    print(f"Avg confidence:       {df['confidence'].mean():.3f}")

    # Naive momentum baseline
    naive_rows = []
    for i in range(1, len(history) - 1):
        prev_delta = history[i]["total_gex"] - history[i - 1]["total_gex"]
        actual_delta = history[i + 1]["total_gex"] - history[i]["total_gex"]
        if prev_delta == 0 and actual_delta == 0:
            correct = 1
        elif prev_delta == 0 or actual_delta == 0:
            correct = 0
        else:
            correct = int(np.sign(prev_delta) == np.sign(actual_delta))
        naive_rows.append(correct)
    if naive_rows:
        print(f"Naive momentum sign:  {np.mean(naive_rows):.3f}")

    # Regime flip recall
    flips = []
    for i in range(len(history) - 1):
        before = history[i]["total_gex"]
        after = history[i + 1]["total_gex"]
        if (before >= 0) != (after >= 0):
            flips.append(i + 1)
    if flips:
        caught = 0
        for flip_idx in flips:
            if flip_idx >= 5:
                pred = predict_next_snapshot(history[:flip_idx])
                if pred and pred.get("regime_flip_probability", 0) > 0.3:
                    caught += 1
        print(f"Regime flip recall:   {caught}/{len(flips)} (prob > 0.3)")

    print("\nLast 5 predictions:")
    print(df[["ts", "actual_delta", "predicted_delta", "delta_sign_correct", "confidence"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    backtest(args.ticker.upper())
