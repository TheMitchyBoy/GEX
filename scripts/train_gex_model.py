"""Train a simple model to predict next-day direction using historical GEX exports.

Usage:
    python scripts/train_gex_model.py --ticker SPX

Requirements:
    - data/exports should contain timestamped CSVs produced by main.py
    - internet access for price history via yfinance

This script:
- Scans `data/exports` for matching CSV sets per timestamp
- Builds simple features from GEX files
- Fetches underlying close prices at timestamps and next-day close
- Trains a logistic regression classifier and saves model to `models/{ticker}_gex_model.joblib`
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
from sklearn.linear_model import LogisticRegression
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from datetime import datetime, timedelta

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
    # keep only timestamps that have at least gex_by_strike and gex_surface
    filtered = {ts: info for ts, info in records.items() if "gex_by_strike" in info}
    return filtered


def parse_timestamp(ts_str: str):
    return datetime.strptime(ts_str, "%Y-%m-%d_%H%M%S")


def compute_features_from_exports(info):
    # info is dict of kind->Path
    features = {}
    # gex by strike
    if "gex_by_strike" in info:
        df = pd.read_csv(info["gex_by_strike"], index_col=0)
        # assume first column is gex in Bn$ / % or similar
        vals = df.iloc[:, 0].astype(float)
        features["total_gex_bn"] = vals.sum()
        features["pos_gex_bn"] = vals[vals > 0].sum()
        features["neg_gex_bn"] = vals[vals < 0].sum()
        features["gex_mean_bn"] = vals.mean()
        features["gex_std_bn"] = vals.std()
        # top 3 magnitudes
        mag = vals.abs().sort_values(ascending=False)
        for i in range(3):
            features[f"top_gex_{i+1}"] = mag.iloc[i] if i < len(mag) else 0.0

    # expiration
    if "gex_by_expiration" in info:
        df = pd.read_csv(info["gex_by_expiration"], index_col=0)
        vals = df.iloc[:, 0].astype(float)
        features["term_total_gex_bn"] = vals.sum()
        # near-term (first 3 expirations) contribution
        features["near_term_gex_bn"] = vals.head(3).sum() if len(vals) > 0 else 0.0

    # surface stats
    if "gex_surface" in info:
        df = pd.read_csv(info["gex_surface"], parse_dates=["expiration"]) if "expiration" in pd.read_csv(info["gex_surface"], nrows=1).columns else pd.read_csv(info["gex_surface"]) 
        if not df.empty and "GEX" in df.columns:
            g = df["GEX"].astype(float)
            features["surface_mean_m"] = g.mean()
            features["surface_std_m"] = g.std()
            features["surface_max_m"] = g.max()
        else:
            features["surface_mean_m"] = 0.0
            features["surface_std_m"] = 0.0
            features["surface_max_m"] = 0.0

    return features


def fetch_price_on_datetime(ticker, dt: datetime):
    # Fetch daily close price for the date containing dt
    # Use yfinance to get history for a small window
    start = (dt - timedelta(days=2)).date().isoformat()
    end = (dt + timedelta(days=3)).date().isoformat()
    # Map common index tickers to Yahoo formats
    yf_symbol = ticker
    if ticker.upper() == "SPX":
        yf_symbol = "^SPX"
    hist = yf.Ticker(yf_symbol).history(start=start, end=end, auto_adjust=False)
    if hist.empty:
        return None
    # find closest previous trading date to dt
    hist.index = pd.to_datetime(hist.index)
    # get last available close at or before dt
    mask = hist.index <= pd.to_datetime(dt)
    if mask.any():
        row = hist.loc[mask].iloc[-1]
        return float(row["Close"])
    else:
        return float(hist.iloc[0]["Close"])


def build_dataset(ticker):
    exports = find_exports_for_ticker(ticker)
    rows = []
    for ts, info in exports.items():
        ts_dt = parse_timestamp(ts)
        features = compute_features_from_exports(info)
        # fetch price at ts and next trading day close
        price = fetch_price_on_datetime(ticker, ts_dt)
        if price is None:
            continue
        # next day
        price_next = fetch_price_on_datetime(ticker, ts_dt + timedelta(days=1))
        if price_next is None:
            continue
        ret = (price_next - price) / price
        label = 1 if ret > 0 else 0
        row = {"ts": ts, "price": price, "price_next": price_next, "ret": ret, "label": label}
        row.update(features)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def train_model(ticker):
    print(f"Building dataset for {ticker}...")
    df = build_dataset(ticker)
    if df.empty:
        print("No data assembled. Ensure exports and internet access are available.")
        return
    # drop non-numeric feature columns and prepare supervised dataset
    feature_df = df.select_dtypes(include=["number"]).drop(columns=["price", "price_next", "ret", "label"]).fillna(0)
    y = df["label"].astype(int)

    # Create lag features (include previous N snapshots' features)
    LAG = 3
    X_lagged = []
    for lag in range(LAG + 1):
        shifted = feature_df.shift(lag).copy()
        shifted.columns = [f"{c}_lag{lag}" for c in shifted.columns]
        X_lagged.append(shifted)
    X_all = pd.concat(X_lagged, axis=1).dropna()

    # Align labels with available rows
    y_aligned = y.loc[X_all.index]

    if y_aligned.nunique() < 2:
        print("Not enough label variety to train a model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_aligned, test_size=0.2, random_state=42, stratify=y_aligned
    )

    # Choose model: XGBoost if available otherwise fallback to LogisticRegression
    model_choice = "xgb" if XGBOOST_AVAILABLE else "logistic"
    print(f"Training model: {model_choice}")

    if XGBOOST_AVAILABLE:
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=4)
    else:
        clf = LogisticRegression(max_iter=400)

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    model_path = MODELS_DIR / f"{ticker}_gex_model.joblib"
    joblib.dump({"model": clf, "features": list(X_all.columns)}, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    train_model(args.ticker.upper())
