"""Backtest simple signals derived from historical GEX exports.

This script:
- Scans `data/exports` for timestamped CSVs for a ticker
- Builds a time-ordered feature table (total_gex, gamma_flip, near_term_ratio)
- Labels each timestamp with next-day return using `yfinance`
- Tests a naive signal: sign(change in total_gex) predicting next-day direction
- Reports accuracy and a simple PnL assuming daily rebalancing

Usage:
    python scripts/backtest_features.py --ticker SPX
"""
import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

try:
    import yfinance as yf
except ModuleNotFoundError:
    print("Missing dependency 'yfinance'. Install with: pip install -r requirements.txt")
    sys.exit(1)

EXPORT_DIR = Path("data/exports")

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
    return records


def parse_ts(ts: str):
    return datetime.strptime(ts, "%Y-%m-%d_%H%M%S")


def compute_gamma_flip(cumulative_csv_path: Path):
    try:
        s = pd.read_csv(cumulative_csv_path, index_col=0, squeeze=True)
    except Exception:
        s = pd.read_csv(cumulative_csv_path, index_col=0).iloc[:, 0]
    s = s.sort_index()
    cum = s.cumsum() if s.name is None else s
    # find zero crossing
    signs = np.sign(cum.values)
    change_points = np.where(np.diff(signs) != 0)[0]
    if len(change_points) == 0:
        return None
    idx = change_points[0]
    x0 = cum.index[idx]
    return float(x0)


def fetch_close_on_date(ticker: str, dt: datetime):
    yf_symbol = "^SPX" if ticker.upper() == "SPX" else ticker
    start = (dt - timedelta(days=3)).date().isoformat()
    end = (dt + timedelta(days=4)).date().isoformat()
    hist = yf.Ticker(yf_symbol).history(start=start, end=end, auto_adjust=False)
    if hist.empty:
        return None
    hist.index = pd.to_datetime(hist.index)
    mask = hist.index <= pd.to_datetime(dt)
    if mask.any():
        return float(hist.loc[mask].iloc[-1]["Close"])
    return float(hist.iloc[0]["Close"])


def build_feature_table(ticker: str):
    exports = find_exports_for_ticker(ticker)
    rows = []
    for ts, info in exports.items():
        # require gex_by_strike to compute total
        if "gex_by_strike" not in info:
            continue
        ts_dt = parse_ts(ts)
        # total gex
        gex = pd.read_csv(info["gex_by_strike"], index_col=0).iloc[:, 0].astype(float)
        total_gex = gex.sum()
        # near_term ratio
        term_total = None
        if "gex_by_expiration" in info:
            gexp = pd.read_csv(info["gex_by_expiration"], index_col=0).iloc[:, 0].astype(float)
            term_total = gexp.sum()
            near_term = gexp.head(3).sum() if len(gexp) > 0 else 0.0
            near_term_ratio = near_term / term_total if term_total != 0 else 0.0
        else:
            near_term_ratio = 0.0

        # gamma flip if cumulative available
        gamma_flip = None
        if "cumulative_gex" in info:
            try:
                cum = pd.read_csv(info["cumulative_gex"], index_col=0).iloc[:, 0].astype(float)
                # find zero crossing
                signs = np.sign(cum.values)
                change_points = np.where(np.diff(signs) != 0)[0]
                if len(change_points) > 0:
                    gamma_flip = float(cum.index[change_points[0]])
            except Exception:
                gamma_flip = None

        rows.append({
            "ts": ts_dt,
            "total_gex": total_gex,
            "near_term_ratio": near_term_ratio,
            "gamma_flip": gamma_flip,
            "ts_str": ts,
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df


def backtest_naive(ticker: str):
    df = build_feature_table(ticker)
    if df.empty:
        print("No export history found for ticker", ticker)
        return
    # fetch prices and labels
    prices = []
    for i, row in df.iterrows():
        p = fetch_close_on_date(ticker, row["ts"])
        pn = fetch_close_on_date(ticker, row["ts"] + timedelta(days=1))
        prices.append((p, pn))
    df["price"] = [p for p, pn in prices]
    df["price_next"] = [pn for p, pn in prices]
    df = df.dropna()
    df["ret_next"] = (df["price_next"] - df["price"]) / df["price"]
    df["label"] = (df["ret_next"] > 0).astype(int)

    # compute signal: sign of change in total_gex from previous timestamp
    df["total_gex_prev"] = df["total_gex"].shift(1)
    df = df.dropna().reset_index(drop=True)
    df["gex_change"] = df["total_gex"] - df["total_gex_prev"]
    df["signal_up"] = (df["gex_change"] > 0).astype(int)

    # evaluate
    accuracy = (df["signal_up"] == df["label"]).mean()
    print(f"Naive signal accuracy (sign of total_gex change): {accuracy:.3f} on {len(df)} samples")

    # simple PnL assuming $1 notional per position and long/short daily
    df["pnl"] = df.apply(lambda r: r["ret_next"] if r["signal_up"] == 1 else -r["ret_next"], axis=1)
    cumulative = (1 + df["pnl"]).cumprod() - 1
    total_return = cumulative.iloc[-1] if len(cumulative) > 0 else 0.0
    print(f"Naive strategy total return (cumulative): {total_return:.4f}")

    # basic report
    print(df[["ts", "total_gex", "gex_change", "signal_up", "label", "ret_next", "pnl"]].tail(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    backtest_naive(args.ticker.upper())
