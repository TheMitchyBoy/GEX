"""Train a model to predict next-snapshot ΔGEX using historical GEX exports.

Usage:
    python scripts/train_gex_model.py --ticker SPX

Saves model to models/{ticker}_gex_delta_model.joblib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

from gex_core.exports import EXPORT_DIR, find_exports_for_ticker, parse_timestamp
from gex_core.features import compute_features_from_exports

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
LAG = 3


def build_dataset(ticker: str) -> pd.DataFrame:
    exports = find_exports_for_ticker(ticker)
    timestamps = sorted(ts for ts, k in exports.items() if "gex_by_strike" in k)
    rows = []
    prev_feats = None
    for ts in timestamps:
        info = exports[ts]
        feats = compute_features_from_exports(info, prev_features=prev_feats)
        prev_feats = feats
        row = {"ts": ts, "ts_dt": parse_timestamp(ts)}
        row.update(feats)
        rows.append(row)

    if len(rows) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("ts_dt").reset_index(drop=True)
    df["target_delta_gex"] = df["total_gex_bn"].shift(-1) - df["total_gex_bn"]
    df["target_total_gex"] = df["total_gex_bn"].shift(-1)
    df = df.dropna(subset=["target_delta_gex"])
    return df


def train_model(ticker: str):
    print(f"Building ΔGEX dataset for {ticker}...")
    df = build_dataset(ticker)
    if df.empty or len(df) < 4:
        print("Not enough export history. Need at least 4 snapshots.")
        return

    feature_cols = [
        c for c in df.columns
        if c not in {"ts", "ts_dt", "target_delta_gex", "target_total_gex", "spot"}
        and df[c].dtype in [np.float64, np.int64, float, int]
    ]
    feature_df = df[feature_cols].fillna(0)

    X_lagged = []
    for lag in range(LAG + 1):
        shifted = feature_df.shift(lag).copy()
        shifted.columns = [f"{c}_lag{lag}" for c in shifted.columns]
        X_lagged.append(shifted)
    X_all = pd.concat(X_lagged, axis=1).dropna()
    y = df.loc[X_all.index, "target_delta_gex"]

    if len(X_all) < 4:
        print("Not enough lagged samples.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42
    )

    model_choice = "xgb" if XGBOOST_AVAILABLE else "linear"
    print(f"Training ΔGEX regressor: {model_choice}")

    if XGBOOST_AVAILABLE:
        reg = xgb.XGBRegressor(n_estimators=100, max_depth=4, n_jobs=4, random_state=42)
    else:
        reg = LinearRegression()

    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    sign_acc = np.mean(np.sign(preds) == np.sign(y_test.values))
    print(f"Test MAE:  {mean_absolute_error(y_test, preds):.4f} Bn$ / %")
    print(f"Test R²:   {r2_score(y_test, preds):.4f}")
    print(f"Sign acc:  {sign_acc:.3f}")

    model_path = MODELS_DIR / f"{ticker}_gex_delta_model.joblib"
    joblib.dump({"model": reg, "features": list(X_all.columns), "target": "delta_gex"}, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    train_model(args.ticker.upper())
