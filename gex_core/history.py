"""Build snapshot history from CSV exports (no database)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gex_core.exports import (
    EXPORT_DIR,
    find_exports_for_ticker,
    load_cumulative_series,
    load_expiration_series,
    load_strike_series,
    load_surface_df,
    parse_timestamp,
)
from gex_core.features import enrich_snapshot_metrics, estimate_gamma_flip


def ts_label(ts: str) -> str:
    return parse_timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def get_latest_ts(ticker: str, export_dir: Path | None = None) -> str | None:
    exports = find_exports_for_ticker(ticker, export_dir)
    if not exports:
        return None
    return max(exports.keys())


def list_timestamps(ticker: str, export_dir: Path | None = None) -> list[str]:
    return sorted(collect_snapshot_files(ticker, export_dir).keys())


def list_tickers(export_dir: Path | None = None) -> list[str]:
    export_dir = export_dir or EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)
    tickers: set[str] = set()
    for path in export_dir.glob("*.csv"):
        parts = path.name.split("_")
        if parts:
            tickers.add(parts[0].upper())
    return sorted(tickers)


def load_snapshot_metrics(ts: str, files: dict[str, Path]) -> dict:
    strike = load_strike_series(files["gex_by_strike"])
    cumulative = (
        load_cumulative_series(files["cumulative_gex"])
        if "cumulative_gex" in files
        else strike.cumsum()
    )

    exp_vals = pd.Series(dtype=float)
    if "gex_by_expiration" in files:
        exp_vals = load_expiration_series(files["gex_by_expiration"])

    surface_df = pd.DataFrame()
    surface_peak = 0.0
    if "gex_surface" in files:
        surface_df = load_surface_df(files["gex_surface"])
        if "GEX" in surface_df.columns and not surface_df.empty:
            surface_peak = float(pd.to_numeric(surface_df["GEX"], errors="coerce").abs().max())

    total_gex = float(strike.sum())
    pos_gex = float(strike[strike > 0].sum())
    neg_gex = float(strike[strike < 0].sum())
    gex_std = float(strike.std()) if len(strike) > 1 else 0.0
    call_wall = float(strike.idxmax()) if len(strike) else None
    put_wall = float(strike.idxmin()) if len(strike) else None
    gamma_flip = estimate_gamma_flip(cumulative)

    near_term = float(exp_vals.head(3).sum()) if len(exp_vals) else 0.0
    term_total = float(exp_vals.sum()) if len(exp_vals) else 0.0
    near_term_ratio = near_term / term_total if term_total else 0.0

    summary_path = files.get("summary")
    source = None
    spot = None
    if summary_path and summary_path.exists():
        import json

        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
        source = summary.get("data_source") or summary.get("source")
        spot = summary.get("spot") or summary.get("spot_price")

    metrics = {
        "ts": ts,
        "ts_label": ts_label(ts),
        "strike": strike,
        "cumulative": cumulative,
        "surface_df": surface_df,
        "surface_path": files.get("gex_surface"),
        "strike_path": files.get("gex_by_strike"),
        "exp_path": files.get("gex_by_expiration"),
        "cum_path": files.get("cumulative_gex"),
        "total_gex": total_gex,
        "pos_gex": pos_gex,
        "neg_gex": neg_gex,
        "gex_std": gex_std,
        "abs_mean": float(strike.abs().mean()) if len(strike) else 0.0,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": gamma_flip,
        "near_term_ratio": near_term_ratio,
        "surface_peak": surface_peak,
        "regime": "LONG gamma" if total_gex >= 0 else "SHORT gamma",
        "data_source": source,
        "spot": spot,
    }
    return enrich_snapshot_metrics(metrics)


def collect_snapshot_files(ticker: str, export_dir: Path | None = None) -> dict[str, dict[str, Path]]:
    """Return {ts: {kind: path}} for snapshots with at least strike + cumulative."""
    export_dir = export_dir or EXPORT_DIR
    exports = find_exports_for_ticker(ticker, export_dir)
    filtered = {}
    for ts, kinds in sorted(exports.items()):
        if "gex_by_strike" not in kinds:
            continue
        if "cumulative_gex" not in kinds:
            kinds = dict(kinds)
            kinds.setdefault("cumulative_gex", kinds["gex_by_strike"])
        summary = export_dir / f"{ticker.upper()}_summary_{ts}.json"
        if summary.exists():
            kinds = dict(kinds)
            kinds["summary"] = summary
        filtered[ts] = kinds
    return filtered


def build_history(ticker: str, export_dir: Path | None = None) -> list[dict]:
    ticker = ticker.upper()
    snapshots = []
    for ts, files in collect_snapshot_files(ticker, export_dir).items():
        try:
            metrics = load_snapshot_metrics(ts, files)
            metrics["ticker"] = ticker
            snapshots.append(metrics)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Skipping snapshot %s for %s: %s", ts, ticker, exc
            )
    snapshots.sort(key=lambda row: row["ts"])
    return snapshots
