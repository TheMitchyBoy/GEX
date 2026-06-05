"""Build snapshot history from CSV exports with optional SQLite index.

``build_history`` joins strike, cumulative, expiration, surface, and summary
files into a chronologically sorted list of snapshot dicts. That list is the
shared memory model for prediction (KNN features), backtests, and dashboard
panels. Enrichment (gamma flip, term structure, concentration) happens in
``gex_core.features.enrich_snapshot_metrics``.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

from gex_core.exports import (
    EXPORT_DIR,
    filter_export_timestamps,
    find_exports_for_ticker,
    list_export_timestamps,
    load_cumulative_series,
    load_expiration_series,
    load_strike_series,
    load_surface_df,
    parse_timestamp,
    paths_for_export_timestamp,
)
from gex_core.features import enrich_snapshot_metrics, estimate_gamma_flip, term_structure_breakdown
from gex_core.tickers import SUPPORTED_TICKERS

_HISTORY_CACHE: dict[tuple, list[dict]] = {}

def _history_limits(
    lookback_days: int | None,
    max_snapshots: int | None,
) -> tuple[int | None, int | None]:
    lb = int(os.environ.get("GEX_HISTORY_LOOKBACK_DAYS", "7")) if lookback_days is None else lookback_days
    cap = int(os.environ.get("GEX_HISTORY_MAX_SNAPSHOTS", "500")) if max_snapshots is None else max_snapshots
    return (None if lb <= 0 else lb), (None if cap <= 0 else cap)


def _history_cache_key(
    ticker: str,
    export_dir: Path | None,
    *,
    lookback_days: int | None,
    max_snapshots: int | None,
) -> tuple:
    export_dir = export_dir or EXPORT_DIR
    timestamps = list_export_timestamps(ticker.upper(), export_dir)
    if not timestamps:
        return (ticker.upper(), str(export_dir.resolve()), "", 0, lookback_days or 0, max_snapshots or 0)
    return (
        ticker.upper(),
        str(export_dir.resolve()),
        timestamps[-1],
        len(timestamps),
        lookback_days or 0,
        max_snapshots or 0,
    )


def clear_history_cache() -> None:
    _HISTORY_CACHE.clear()
    build_history_cached.cache_clear()


@lru_cache(maxsize=16)
def build_history_cached(
    ticker: str,
    export_dir_str: str,
    sig_ts: str,
    sig_n: int,
    lookback_days: int,
    max_snapshots: int,
) -> tuple:
    """LRU-cached history builder; returns tuple for hashability."""
    return tuple(
        _build_history_impl(
            ticker,
            Path(export_dir_str),
            lookback_days=lookback_days or None,
            max_snapshots=max_snapshots or None,
        )
    )


def ts_label(ts: str) -> str:
    return parse_timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def get_latest_ts(ticker: str, export_dir: Path | None = None) -> str | None:
    export_dir = export_dir or EXPORT_DIR
    if export_dir.resolve() != EXPORT_DIR.resolve():
        timestamps = list_export_timestamps(ticker, export_dir)
        return timestamps[-1] if timestamps else None

    from gex_core.storage import latest_timestamp

    ts = latest_timestamp(ticker, export_dir)
    if ts:
        return ts
    timestamps = list_export_timestamps(ticker, export_dir)
    return timestamps[-1] if timestamps else None


def list_timestamps(ticker: str, export_dir: Path | None = None) -> list[str]:
    from gex_core.storage import list_indexed_timestamps, sync_ticker_exports

    export_dir = export_dir or EXPORT_DIR
    try:
        sync_ticker_exports(ticker, export_dir)
        indexed = list_indexed_timestamps(ticker)
        if indexed:
            return indexed
    except Exception as exc:
        logging.getLogger(__name__).debug("Index timestamp list unavailable: %s", exc)
    return sorted(collect_snapshot_files(ticker, export_dir).keys())


def list_tickers(export_dir: Path | None = None) -> list[str]:
    export_dir = export_dir or EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)
    tickers: set[str] = set()
    for path in export_dir.glob("*.csv"):
        parts = path.name.split("_")
        if parts:
            ticker = parts[0].upper()
            if ticker in SUPPORTED_TICKERS:
                tickers.add(ticker)
    return [ticker for ticker in SUPPORTED_TICKERS if ticker in tickers]


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

    term_breakdown = term_structure_breakdown(
        exp_vals,
        snapshot_date=pd.Timestamp(parse_timestamp(ts)),
    )

    summary_path = files.get("summary")
    source = None
    spot = None
    extended_features: dict = {}
    if summary_path and summary_path.exists():
        import json

        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
        source = summary.get("data_source") or summary.get("source")
        spot = summary.get("spot") or summary.get("spot_price")
        extended_features = summary.get("extended_features") or {}

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
        **term_breakdown,
        "surface_peak": surface_peak,
        "regime": "LONG gamma" if total_gex >= 0 else "SHORT gamma",
        "data_source": source,
        "spot": spot,
        "extended_features": extended_features,
    }
    metrics.update({k: float(v) for k, v in extended_features.items() if isinstance(v, (int, float))})
    return enrich_snapshot_metrics(metrics)


def collect_snapshot_files(
    ticker: str,
    export_dir: Path | None = None,
    *,
    lookback_days: int | None = None,
    max_snapshots: int | None = None,
) -> dict[str, dict[str, Path]]:
    """Return {ts: {kind: path}} for snapshots with at least strike + cumulative."""
    export_dir = export_dir or EXPORT_DIR
    timestamps = list_export_timestamps(ticker.upper(), export_dir)
    if lookback_days and lookback_days > 0 and timestamps:
        latest = parse_timestamp(timestamps[-1])
        since = latest - timedelta(days=lookback_days)
        timestamps = filter_export_timestamps(timestamps, since=since, max_timestamps=max_snapshots)
    elif max_snapshots:
        timestamps = filter_export_timestamps(timestamps, max_timestamps=max_snapshots)

    exports = {
        ts: paths_for_export_timestamp(ticker, ts, export_dir)
        for ts in timestamps
    }
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


def _build_history_impl(
    ticker: str,
    export_dir: Path,
    *,
    lookback_days: int | None = None,
    max_snapshots: int | None = None,
) -> list[dict]:
    ticker = ticker.upper()
    snapshots = []
    for ts, files in collect_snapshot_files(
        ticker,
        export_dir,
        lookback_days=lookback_days,
        max_snapshots=max_snapshots,
    ).items():
        try:
            metrics = load_snapshot_metrics(ts, files)
            metrics["ticker"] = ticker
            snapshots.append(metrics)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Skipping snapshot %s for %s: %s", ts, ticker, exc
            )
    snapshots.sort(key=lambda row: row["ts"])
    deduped = []
    for row in snapshots:
        if deduped and row.get("strike") is not None and row["strike"].equals(deduped[-1].get("strike")):
            logging.getLogger(__name__).info(
                "Skipping duplicate snapshot %s for %s; strike profile matches %s",
                row["ts"],
                ticker,
                deduped[-1]["ts"],
            )
            continue
        deduped.append(row)
    return deduped


def build_history(
    ticker: str,
    export_dir: Path | None = None,
    *,
    lookback_days: int | None = None,
    max_snapshots: int | None = None,
) -> list[dict]:
    export_dir = export_dir or EXPORT_DIR
    lb, cap = _history_limits(lookback_days, max_snapshots)
    key = _history_cache_key(ticker, export_dir, lookback_days=lb, max_snapshots=cap)
    if key in _HISTORY_CACHE:
        return _HISTORY_CACHE[key]
    cached = build_history_cached(key[0], key[1], key[2], key[3], key[4], key[5])
    history = list(cached)
    _HISTORY_CACHE[key] = history
    return history
