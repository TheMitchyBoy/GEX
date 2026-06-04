"""Train an LSTM to predict next-snapshot ΔGEX from sequential GEX features.

Usage:
    python scripts/train_gex_lstm.py --ticker SPX --seq-len 8

Saves model to models/{ticker}_gex_lstm/ and meta to models/{ticker}_gex_lstm/meta.joblib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

from gex_core.exports import find_exports_for_ticker, parse_timestamp
from gex_core.features import compute_features_from_exports

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def build_feature_timeseries(ticker: str) -> pd.DataFrame:
    exports = find_exports_for_ticker(ticker)
    timestamps = sorted(ts for ts, k in exports.items() if "gex_by_strike" in k)
    rows = []
    prev_feats = None
    for ts in timestamps:
        info = exports[ts]
        feats = compute_features_from_exports(info, prev_features=prev_feats)
        prev_feats = feats
        row = {"ts": parse_timestamp(ts)}
        row.update(feats)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    df["target_delta_gex"] = df["total_gex_bn"].shift(-1) - df["total_gex_bn"]
    df = df.dropna(subset=["target_delta_gex"])
    df.set_index("ts", inplace=True)
    return df


def create_sequences(df: pd.DataFrame, seq_len: int = 8):
    exclude = {"target_delta_gex", "spot"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]
    X, y = [], []
    for i in range(len(df) - seq_len):
        seq = df.iloc[i : i + seq_len][feature_cols].values
        label = df.iloc[i + seq_len]["target_delta_gex"]
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y), feature_cols


def train_lstm(ticker: str, seq_len: int = 8, epochs: int = 50, batch_size: int = 16):
    print(f"Building ΔGEX sequence dataset for {ticker}...")
    df = build_feature_timeseries(ticker)
    if df.empty:
        print("No data available.")
        return

    X, y, feature_cols = create_sequences(df, seq_len=seq_len)
    if X.size == 0:
        print(f"Need more snapshots (seq_len={seq_len}).")
        return

    split = max(int(0.8 * len(X)), 1)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    n_features = X.shape[2]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape) if len(X_test) else X_test

    tf.random.set_seed(42)
    model = Sequential([
        LSTM(64, input_shape=(seq_len, n_features), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    val_data = (X_test_scaled, y_test) if len(X_test) else None
    model.fit(
        X_train_scaled, y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early],
        verbose=2,
    )

    if len(X_test):
        preds = model.predict(X_test_scaled, verbose=0).flatten()
        mae = float(np.mean(np.abs(preds - y_test)))
        sign_acc = float(np.mean(np.sign(preds) == np.sign(y_test)))
        print(f"Test MAE: {mae:.4f} Bn$ / %, sign accuracy: {sign_acc:.3f}")

    model_path = MODELS_DIR / f"{ticker}_gex_lstm.keras"
    model.save(str(model_path))
    joblib.dump(
        {"scaler": scaler, "meta": {"features": feature_cols, "seq_len": seq_len, "target": "delta_gex"}},
        MODELS_DIR / f"{ticker}_gex_lstm_meta.joblib",
    )
    print(f"Saved LSTM model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train_lstm(args.ticker.upper(), seq_len=args.seq_len, epochs=args.epochs, batch_size=args.batch_size)
