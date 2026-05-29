"""Train an LSTM model on historical GEX exports to predict next-day direction.

Usage:
    python scripts/train_gex_lstm.py --ticker SPX --seq-len 8

Notes:
- Requires `data/exports` CSV snapshots and internet access for price labels via yfinance.
- Saves model to `models/{ticker}_gex_lstm/` and scaler/features to `models/{ticker}_gex_lstm_meta.joblib`.
"""
import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
import sys
try:
    import yfinance as yf
except ModuleNotFoundError:
    print("Missing dependency 'yfinance'. Install with: pip install -r requirements.txt")
    sys.exit(1)
from datetime import datetime, timedelta
import joblib

# Keras / TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

EXPORT_DIR = Path("data/exports")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TIMESTAMP_RE = re.compile(r"^(?P<ticker>[A-Z0-9]+)_(?P<kind>gex_by_strike|gex_by_expiration|gex_surface|cumulative_gex)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{6})\.csv$")


def find_exports_for_ticker(ticker: str):
    files = list(EXPORT_DIR.glob(f"{ticker}_*_*_*.csv"))
    records = {}
    for f in files:
        m = TIMESTAMP_RE.match(f.name)
        if not m:
            continue
        ts = m.group("ts")
        kind = m.group("kind")
        records.setdefault(ts, {})[kind] = f
    # Keep entries that have at least gex_by_strike
    filtered = {ts: info for ts, info in records.items() if "gex_by_strike" in info}
    return filtered


def parse_timestamp(ts_str: str):
    return datetime.strptime(ts_str, "%Y-%m-%d_%H%M%S")


def compute_features_from_exports(info):
    features = {}
    # gex by strike
    if "gex_by_strike" in info:
        df = pd.read_csv(info["gex_by_strike"], index_col=0)
        vals = df.iloc[:, 0].astype(float)
        features["total_gex_bn"] = vals.sum()
        features["pos_gex_bn"] = vals[vals > 0].sum()
        features["neg_gex_bn"] = vals[vals < 0].sum()
        features["gex_mean_bn"] = vals.mean()
        features["gex_std_bn"] = vals.std()
    else:
        # default zeros
        features.update({k: 0.0 for k in ["total_gex_bn","pos_gex_bn","neg_gex_bn","gex_mean_bn","gex_std_bn"]})

    # expiration
    if "gex_by_expiration" in info:
        df = pd.read_csv(info["gex_by_expiration"], index_col=0)
        vals = df.iloc[:, 0].astype(float)
        features["term_total_gex_bn"] = vals.sum()
        features["near_term_gex_bn"] = vals.head(3).sum() if len(vals) > 0 else 0.0
    else:
        features.update({"term_total_gex_bn": 0.0, "near_term_gex_bn": 0.0})

    # surface
    if "gex_surface" in info:
        try:
            df = pd.read_csv(info["gex_surface"], parse_dates=["expiration"]) if "expiration" in pd.read_csv(info["gex_surface"], nrows=1).columns else pd.read_csv(info["gex_surface"]) 
            if not df.empty and "GEX" in df.columns:
                g = df["GEX"].astype(float)
                features["surface_mean_m"] = g.mean()
                features["surface_std_m"] = g.std()
            else:
                features["surface_mean_m"] = 0.0
                features["surface_std_m"] = 0.0
        except Exception:
            features["surface_mean_m"] = 0.0
            features["surface_std_m"] = 0.0
    else:
        features["surface_mean_m"] = 0.0
        features["surface_std_m"] = 0.0

    return features


def fetch_price_on_datetime(ticker, dt: datetime):
    yf_symbol = ticker
    if ticker.upper() == "SPX":
        yf_symbol = "^SPX"
    start = (dt - timedelta(days=2)).date().isoformat()
    end = (dt + timedelta(days=3)).date().isoformat()
    hist = yf.Ticker(yf_symbol).history(start=start, end=end, auto_adjust=False)
    if hist.empty:
        return None
    hist.index = pd.to_datetime(hist.index)
    mask = hist.index <= pd.to_datetime(dt)
    if mask.any():
        row = hist.loc[mask].iloc[-1]
        return float(row["Close"])
    else:
        return float(hist.iloc[0]["Close"])


def build_feature_timeseries(ticker: str):
    exports = find_exports_for_ticker(ticker)
    rows = []
    for ts, info in exports.items():
        ts_dt = parse_timestamp(ts)
        features = compute_features_from_exports(info)
        price = fetch_price_on_datetime(ticker, ts_dt)
        if price is None:
            continue
        price_next = fetch_price_on_datetime(ticker, ts_dt + timedelta(days=1))
        if price_next is None:
            continue
        ret = (price_next - price) / price
        label = 1 if ret > 0 else 0
        row = {"ts": ts_dt, "price": price, "price_next": price_next, "ret": ret, "label": label}
        row.update(features)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(by="ts").reset_index(drop=True)
    df.set_index("ts", inplace=True)
    return df


def create_sequences(df: pd.DataFrame, seq_len=8):
    feature_cols = [c for c in df.columns if c not in ["price", "price_next", "ret", "label"]]
    X = []
    y = []
    for i in range(len(df) - seq_len):
        seq = df.iloc[i : i + seq_len][feature_cols].values
        label = df.iloc[i + seq_len]["label"]
        X.append(seq)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y, feature_cols


def train_lstm(ticker: str, seq_len=8, epochs=50, batch_size=16):
    print(f"Building feature timeline for {ticker}...")
    df = build_feature_timeseries(ticker)
    if df.empty:
        print("No data available to train. Ensure exports and internet access.")
        return

    X, y, feature_cols = create_sequences(df, seq_len=seq_len)
    if X.size == 0:
        print("Not enough sequential snapshots to build sequences. Need more exports.")
        return

    # Chronological train/val split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Flatten scaler fit on features
    n_features = X.shape[2]
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    # Build LSTM model
    tf.random.set_seed(42)
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_len, n_features), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early],
        verbose=2,
    )

    # Evaluate
    loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}, loss: {loss:.4f}")

    # Save model and metadata
    model_dir = MODELS_DIR / f"{ticker}_gex_lstm"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))

    meta = {"features": feature_cols, "seq_len": seq_len}
    joblib.dump({"scaler": scaler, "meta": meta}, model_dir / "meta.joblib")

    print(f"Saved LSTM model to {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train_lstm(args.ticker.upper(), seq_len=args.seq_len, epochs=args.epochs, batch_size=args.batch_size)
