"""Backtest simple signals derived from historical GEX exports.

This script:
- Scans `data/exports` for timestamped CSVs for a ticker
- Builds a time-ordered feature table (total_gex, gamma_flip, near_term_ratio)
- Labels each timestamp with next-day return using `yfinance`
- Tests a naive signal: sign(change in total_gex) predicting next-day direction
- Reports accuracy and a simple PnL assuming daily rebalancing

For ΔGEX prediction backtesting, see scripts/backtest_gex_prediction.py

Usage:
    python scripts/backtest_features.py --ticker SPX
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ModuleNotFoundError:
    print("Missing dependency 'yfinance'. Install with: pip install -r requirements.txt")
    sys.exit(1)

from gex_core.exports import find_exports_for_ticker, load_strike_series, load_expiration_series, parse_timestamp
from gex_core.features import estimate_gamma_flip


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
        ts_dt = parse_timestamp(ts)
        gex = load_strike_series(info["gex_by_strike"])
        total_gex = float(gex.sum())
        if "gex_by_expiration" in info:
            gexp = load_expiration_series(info["gex_by_expiration"])
            term_total = float(gexp.sum())
            near_term = float(gexp.head(3).sum()) if len(gexp) else 0.0
            near_term_ratio = near_term / term_total if term_total != 0 else 0.0
        else:
            near_term_ratio = 0.0

        gamma_flip = None
        if "cumulative_gex" in info:
            try:
                cum = load_strike_series(info["cumulative_gex"])
                gamma_flip = estimate_gamma_flip(cum)
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
