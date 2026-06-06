"""Export catalog vs on-disk CSV diagnostics for dashboards and health checks."""

from __future__ import annotations

import os
from typing import Any

from gex_core.exports import list_export_timestamps
from gex_core.history import build_history, collect_snapshot_files, load_snapshot_metrics
from gex_core.predict import MIN_KNN_SNAPSHOTS
from gex_core.storage import count_strike_exports_on_disk, list_indexed_timestamps, sync_ticker_exports


def prediction_lookback_days(ticker: str | None = None) -> int:
    """Resolve KNN lookback, auto-widening when the configured window is too thin."""
    configured = int(os.environ.get("GEX_PREDICTION_LOOKBACK_DAYS", "90"))
    fallback = int(os.environ.get("GEX_PREDICTION_LOOKBACK_FALLBACK_DAYS", "90"))
    max_snapshots = int(os.environ.get("GEX_PREDICTION_HISTORY_MAX", "240"))
    if configured >= fallback:
        return configured
    if ticker:
        on_disk = count_strike_exports_on_disk(ticker)
        if on_disk < MIN_KNN_SNAPSHOTS:
            return configured
        in_window = len(
            collect_snapshot_files(
                ticker,
                lookback_days=configured,
                max_snapshots=max_snapshots,
            )
        )
        if in_window >= MIN_KNN_SNAPSHOTS:
            return configured
        wider = len(
            collect_snapshot_files(
                ticker,
                lookback_days=fallback,
                max_snapshots=max_snapshots,
            )
        )
        if wider >= MIN_KNN_SNAPSHOTS:
            return fallback
    return configured


def summarize_export_state(
    ticker: str,
    *,
    lookback_days: int | None = None,
    max_snapshots: int | None = None,
    dedupe_identical_strikes: bool = False,
) -> dict[str, Any]:
    """Compare SQLite index rows, strike CSVs on disk, and loadable snapshot depth."""
    ticker = ticker.upper()
    lookback_days = prediction_lookback_days(ticker) if lookback_days is None else lookback_days
    max_snapshots = int(os.environ.get("GEX_PREDICTION_HISTORY_MAX", "240")) if max_snapshots is None else max_snapshots

    indexed_before = len(list_indexed_timestamps(ticker))
    strike_csv_on_disk = count_strike_exports_on_disk(ticker)
    sync_ticker_exports(ticker)
    indexed_after = len(list_indexed_timestamps(ticker))
    catalog_timestamps = list_export_timestamps(ticker)

    collected = collect_snapshot_files(
        ticker,
        lookback_days=lookback_days,
        max_snapshots=max_snapshots,
    )
    history = build_history(
        ticker,
        lookback_days=lookback_days,
        max_snapshots=max_snapshots,
        dedupe_identical_strikes=dedupe_identical_strikes,
    )
    load_errors: list[str] = []
    if collected and not history:
        for ts, files in list(collected.items())[:3]:
            try:
                load_snapshot_metrics(ts, files)
            except Exception as exc:
                load_errors.append(f"{ts}: {exc}")

    return {
        "indexed_before_sync": indexed_before,
        "indexed_after_sync": indexed_after,
        "strike_csv_on_disk": strike_csv_on_disk,
        "catalog_timestamps": len(catalog_timestamps),
        "collected_in_window": len(collected),
        "metrics_load_ok": len(history),
        "forecast_loadable": len(history),
        "lookback_days": lookback_days,
        "max_snapshots": max_snapshots,
        "sample_load_errors": load_errors,
        "needs_backfill": strike_csv_on_disk < MIN_KNN_SNAPSHOTS,
        "index_stale": indexed_before > strike_csv_on_disk,
    }


def forecast_blocker_from_state(state: dict[str, Any], *, window_count: int) -> str:
    """Build a user-facing forecast blocker from ``summarize_export_state`` output."""
    if window_count >= MIN_KNN_SNAPSHOTS:
        return ""

    disk = int(state.get("strike_csv_on_disk") or 0)
    loaded = int(state.get("forecast_loadable") or window_count)
    indexed_before = int(state.get("indexed_before_sync") or 0)
    in_window = int(state.get("collected_in_window") or 0)
    lookback = state.get("lookback_days")

    msg = (
        f"Need at least {MIN_KNN_SNAPSHOTS} loadable snapshots for KNN; found {window_count} "
        f"in the {lookback}-day forecast window."
    )
    msg += f" Strike CSV files on disk: {disk}."
    if indexed_before > disk:
        msg += (
            f" SQLite index previously listed {indexed_before} timestamps"
            f" ({indexed_before - disk} stale rows without CSV files)."
        )
    if disk < MIN_KNN_SNAPSHOTS:
        msg += (
            " Export history was lost or never backfilled on this server — run a one-time"
            " intraday backfill (Railway: set GEX_STARTUP_BACKFILL=1 and start command"
            " `bash scripts/start_web.sh`, then redeploy)."
        )
    elif in_window > loaded:
        msg += (
            f" {in_window} timestamps matched the lookback window but only {loaded} loaded;"
            " check service logs for CSV parse errors."
        )
    elif in_window < MIN_KNN_SNAPSHOTS and disk >= MIN_KNN_SNAPSHOTS:
        msg += (
            f" Only {in_window} timestamps fall in the last {lookback} days;"
            " raise GEX_PREDICTION_LOOKBACK_DAYS (e.g. 90) or backfill recent data."
        )
    return msg
