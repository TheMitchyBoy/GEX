"""Train a model to predict next-snapshot ΔGEX using historical GEX exports.

Usage:
    python scripts/train_gex_model.py --ticker SPX

Saves model to models/{ticker}_gex_delta_model.joblib
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

from gex_core.exports import EXPORT_DIR, list_export_timestamps, load_strike_series, parse_timestamp, paths_for_export_timestamp
from gex_core.features import compute_features_from_exports

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
LAG = 3
# Wider default window than the original 7 days: a one-week window routinely
# produced only 2-3 training rows. 30 days keeps the model regime-relevant while
# giving the regressor enough samples to be meaningful.
# 0 = use every export on disk (recommended when you have 3000+ intraday snapshots).
DEFAULT_LOOKBACK_DAYS = int(os.environ.get("GEX_TRAIN_LOOKBACK_DAYS", "0"))
# Minimum folds before walk-forward CV metrics are reported.
MIN_CV_FOLDS = 3


def _train_dedupe_identical_strikes() -> bool:
    return os.environ.get("GEX_TRAIN_DEDUPE_IDENTICAL_STRIKES", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _summary_spot(info: dict[str, Path]) -> float | None:
    summary_path = info.get("summary")
    if not summary_path or not summary_path.exists():
        return None
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    spot = summary.get("spot") or summary.get("spot_price")
    return float(spot) if spot is not None else None


def _recent_timestamps(timestamps: list[str], lookback_days: int | None) -> list[str]:
    if not timestamps or not lookback_days or lookback_days <= 0:
        return timestamps
    parsed = pd.to_datetime([parse_timestamp(ts) for ts in timestamps])
    cutoff = parsed.max() - pd.Timedelta(days=lookback_days)
    return [ts for ts, ts_dt in zip(timestamps, parsed) if ts_dt >= cutoff]


def build_dataset(
    ticker: str,
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
    *,
    dedupe_identical_strikes: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build supervised training rows from export CSVs.

    Returns ``(dataframe, stats)`` where stats reports catalog depth vs rows kept.
    """
    catalog = list_export_timestamps(ticker)
    timestamps = _recent_timestamps(catalog, lookback_days)
    dedupe = _train_dedupe_identical_strikes() if dedupe_identical_strikes is None else dedupe_identical_strikes
    rows = []
    skipped_identical = 0
    skipped_missing = 0
    prev_feats = None
    prev_strike = None
    for ts in timestamps:
        info = paths_for_export_timestamp(ticker, ts, EXPORT_DIR)
        strike_path = info.get("gex_by_strike")
        if not strike_path:
            skipped_missing += 1
            continue
        strike = load_strike_series(strike_path)
        if (
            dedupe
            and prev_strike is not None
            and strike is not None
            and strike.equals(prev_strike)
        ):
            skipped_identical += 1
            continue
        feats = compute_features_from_exports(info, spot=_summary_spot(info), prev_features=prev_feats)
        prev_feats = feats
        prev_strike = strike
        row = {"ts": ts, "ts_dt": parse_timestamp(ts)}
        row.update(feats)
        rows.append(row)

    stats = {
        "catalog_timestamps": len(catalog),
        "window_timestamps": len(timestamps),
        "training_rows": len(rows),
        "skipped_identical_strikes": skipped_identical,
        "skipped_missing_strike_csv": skipped_missing,
    }

    if len(rows) < 2:
        return pd.DataFrame(), stats

    df = pd.DataFrame(rows).sort_values("ts_dt").reset_index(drop=True)
    df["target_delta_gex"] = df["total_gex_bn"].shift(-1) - df["total_gex_bn"]
    df["target_total_gex"] = df["total_gex_bn"].shift(-1)
    df = df.dropna(subset=["target_delta_gex"])
    stats["supervised_rows"] = int(len(df))
    return df, stats


def _chronological_split(X_all: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    test_size = max(1, int(np.ceil(len(X_all) * 0.2)))
    if len(X_all) - test_size < 2:
        test_size = 1
    split_at = len(X_all) - test_size
    return X_all.iloc[:split_at], X_all.iloc[split_at:], y.iloc[:split_at], y.iloc[split_at:]


def _make_regressor(n_train_rows: int):
    if XGBOOST_AVAILABLE and n_train_rows >= 10:
        return xgb.XGBRegressor(n_estimators=100, max_depth=4, n_jobs=4, random_state=42), "xgb"
    return LinearRegression(), "linear"


def walk_forward_cv(X_all: pd.DataFrame, y: pd.Series, min_train: int = 4) -> dict:
    """Rolling-origin (expanding window) one-step-ahead cross-validation.

    Far more honest on small samples than a single 80/20 split: every fold
    trains on the past and predicts the next point, then expands. Reports mean
    out-of-sample MAE and sign accuracy across folds.
    """
    max_folds = int(os.environ.get("GEX_TRAIN_MAX_CV_FOLDS", "120"))
    start_at = min_train
    if len(X_all) - min_train > max_folds:
        start_at = len(X_all) - max_folds

    preds, actuals = [], []
    for split_at in range(start_at, len(X_all)):
        X_tr, y_tr = X_all.iloc[:split_at], y.iloc[:split_at]
        X_te, y_te = X_all.iloc[split_at : split_at + 1], y.iloc[split_at : split_at + 1]
        model, _ = _make_regressor(len(X_tr))
        model.fit(X_tr, y_tr)
        preds.append(float(model.predict(X_te)[0]))
        actuals.append(float(y_te.iloc[0]))
    if len(preds) < MIN_CV_FOLDS:
        return {"cv_folds": len(preds), "cv_mae": None, "cv_sign_accuracy": None}
    preds_arr = np.asarray(preds)
    actuals_arr = np.asarray(actuals)
    return {
        "cv_folds": len(preds),
        "cv_mae": float(mean_absolute_error(actuals_arr, preds_arr)),
        "cv_sign_accuracy": float(np.mean(np.sign(preds_arr) == np.sign(actuals_arr))),
    }


def train_model(ticker: str, lookback_days: int | None = DEFAULT_LOOKBACK_DAYS):
    window_label = f"last {lookback_days} days" if lookback_days and lookback_days > 0 else "full export catalog"
    print(f"Building ΔGEX dataset for {ticker} over the {window_label}...")
    df, stats = build_dataset(ticker, lookback_days=lookback_days)
    print(
        f"Catalog: {stats['catalog_timestamps']} timestamps | "
        f"window: {stats['window_timestamps']} | "
        f"training rows: {stats.get('supervised_rows', stats['training_rows'])}"
        + (
            f" | skipped identical strikes: {stats['skipped_identical_strikes']}"
            if stats["skipped_identical_strikes"]
            else ""
        )
    )
    if df.empty or len(df) < 3:
        print("Not enough export history. Need at least 4 loadable snapshots.")
        return

    feature_cols = [
        c for c in df.columns
        if c not in {"ts", "ts_dt", "target_delta_gex", "target_total_gex", "spot"}
        and df[c].dtype in [np.float64, np.int64, float, int]
    ]
    feature_df = df[feature_cols].fillna(0)

    X_lagged = []
    effective_lag = min(LAG, max(0, len(feature_df) - 4))
    for lag in range(effective_lag + 1):
        shifted = feature_df.shift(lag).copy()
        shifted.columns = [f"{c}_lag{lag}" for c in shifted.columns]
        X_lagged.append(shifted)
    X_all = pd.concat(X_lagged, axis=1).dropna()
    y = df.loc[X_all.index, "target_delta_gex"]

    if len(X_all) < 3:
        print("Not enough lagged samples.")
        return

    X_train, X_test, y_train, y_test = _chronological_split(X_all, y)

    reg, model_choice = _make_regressor(len(X_train))
    print(f"Training ΔGEX regressor: {model_choice}")

    cv_metrics = walk_forward_cv(X_all, y)
    if cv_metrics["cv_sign_accuracy"] is not None:
        print(
            f"Walk-forward CV ({cv_metrics['cv_folds']} folds): "
            f"MAE {cv_metrics['cv_mae']:.4f} Bn$ / %, "
            f"sign acc {cv_metrics['cv_sign_accuracy']:.3f}"
        )
    else:
        print("Walk-forward CV: not enough folds to report.")

    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    sign_acc = np.mean(np.sign(preds) == np.sign(y_test.values))
    print(f"Test MAE:  {mean_absolute_error(y_test, preds):.4f} Bn$ / %")
    test_r2 = r2_score(y_test, preds) if len(y_test) > 1 else float("nan")
    print(f"Test R²:   {test_r2:.4f}" if not np.isnan(test_r2) else "Test R²:   N/A")
    print(f"Sign acc:  {sign_acc:.3f}")

    model_path = MODELS_DIR / f"{ticker}_gex_delta_model.joblib"
    joblib.dump({"model": reg, "features": list(X_all.columns), "target": "delta_gex"}, model_path)
    print(f"Saved model to {model_path}")

    from gex_core.models_manifest import write_manifest

    write_manifest(
        ticker,
        model_type=model_choice,
        metrics={
            "test_mae": float(mean_absolute_error(y_test, preds)),
            "test_r2": None if np.isnan(test_r2) else float(test_r2),
            "sign_accuracy": float(sign_acc),
            "n_train": int(len(X_train)),
            "cv_folds": cv_metrics["cv_folds"],
            "cv_mae": cv_metrics["cv_mae"],
            "cv_sign_accuracy": cv_metrics["cv_sign_accuracy"],
        },
        extra={
            "training_start_ts": str(df["ts"].iloc[0]),
            "training_end_ts": str(df["ts"].iloc[-1]),
            "lookback_days": lookback_days,
            "lag": effective_lag,
            "n_snapshots": int(len(df) + 1),
            "catalog_timestamps": stats["catalog_timestamps"],
            "window_timestamps": stats["window_timestamps"],
            "skipped_identical_strikes": stats["skipped_identical_strikes"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Training window in days (default 0 = all export history on disk).",
    )
    parser.add_argument(
        "--dedupe-identical-strikes",
        action="store_true",
        help="Skip consecutive snapshots with identical strike profiles (default: keep all).",
    )
    args = parser.parse_args()
    if args.dedupe_identical_strikes:
        os.environ["GEX_TRAIN_DEDUPE_IDENTICAL_STRIKES"] = "1"
    train_model(args.ticker.upper(), lookback_days=args.lookback_days)
