"""Import on-disk CSV/JSON export sets into PostgreSQL snapshot tables."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from gex_core.exports import (
    EXPORT_DIR,
    find_exports_for_ticker,
    load_cumulative_series,
    load_expiration_series,
    load_greek_exposure_df,
    load_strike_series,
    load_surface_df,
    paths_for_export_timestamp,
    scan_export_timestamps,
)
from gex_core.features import safe_float

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    ticker: str
    ts: str
    status: str
    written: bool = False
    skipped: bool = False
    error: str | None = None


def normalize_legacy_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten older summary JSON shapes before validation/features."""
    out = dict(summary)
    flip = out.get("gamma_flip")
    if isinstance(flip, dict):
        out["gamma_flip"] = flip.get("flip_strike")
    for wall in ("call_wall", "put_wall"):
        value = out.get(wall)
        if isinstance(value, dict):
            out[wall] = value.get("strike")
    if not out.get("market_date") and out.get("ticker"):
        pass
    return out


def load_export_snapshot(
    ticker: str,
    ts: str,
    *,
    export_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Load one export timestamp from disk into write-ready structures."""
    export_dir = export_dir or EXPORT_DIR
    kinds = paths_for_export_timestamp(ticker, ts, export_dir)
    strike_path = kinds.get("gex_by_strike")
    if strike_path is None or not strike_path.exists():
        return None

    gex_by_strike = load_strike_series(strike_path)
    if gex_by_strike.empty:
        return None

    cumulative_path = kinds.get("cumulative_gex")
    if cumulative_path and cumulative_path.exists():
        cumulative_gex = load_cumulative_series(cumulative_path)
        cumulative_gex = cumulative_gex.reindex(gex_by_strike.index)
        cumulative_gex = cumulative_gex.fillna(gex_by_strike.cumsum())
    else:
        cumulative_gex = gex_by_strike.cumsum()

    expiration_path = kinds.get("gex_by_expiration")
    gex_by_expiration = (
        load_expiration_series(expiration_path)
        if expiration_path and expiration_path.exists()
        else pd.Series(dtype=float)
    )

    surface_data = (
        load_surface_df(kinds["gex_surface"])
        if kinds.get("gex_surface") and kinds["gex_surface"].exists()
        else pd.DataFrame()
    )
    greek_exposure_df = (
        load_greek_exposure_df(kinds["greek_exposure"])
        if kinds.get("greek_exposure") and kinds["greek_exposure"].exists()
        else pd.DataFrame()
    )

    summary: dict[str, Any] = {}
    summary_path = kinds.get("summary")
    if summary_path and summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
    summary = normalize_legacy_summary(summary)
    summary.setdefault("data_source", "csv_import")
    summary.setdefault("strike_profile_source", "csv_import")

    spot = safe_float(summary.get("spot") or summary.get("spot_price"), 0.0)
    if spot <= 0 and not gex_by_strike.empty:
        spot = float(pd.to_numeric(gex_by_strike.index, errors="coerce").median())
        summary["spot"] = spot
        summary["spot_price"] = spot

    from gex_core.strike_filter import resolve_storage_strike_profile

    storage_strikes, strike_source = resolve_storage_strike_profile(
        gex_by_strike,
        spot=spot,
        greek_df=greek_exposure_df if not greek_exposure_df.empty else None,
    )
    if len(storage_strikes) != len(gex_by_strike):
        gex_by_strike = storage_strikes
        cumulative_gex = gex_by_strike.cumsum()
        summary["strike_profile_source"] = strike_source
    summary["total_gex_bn_per_pct"] = float(gex_by_strike.sum())
    if not summary.get("net_gamma_regime"):
        summary["net_gamma_regime"] = (
            "LONG gamma" if summary["total_gex_bn_per_pct"] >= 0 else "SHORT gamma"
        )
    if not summary.get("market_date"):
        summary["market_date"] = ts.split("_", 1)[0]

    return {
        "gex_by_strike": gex_by_strike,
        "cumulative_gex": cumulative_gex,
        "gex_by_expiration": gex_by_expiration,
        "surface_data": surface_data,
        "greek_exposure_df": greek_exposure_df,
        "summary": summary,
        "summary_path": str(summary_path) if summary_path else None,
        "strike_path": str(strike_path),
    }


def list_postgres_timestamps(ticker: str) -> set[str]:
    from gex_core.db import database_url, ensure_postgres_schema, use_postgres

    if not use_postgres():
        return set()
    ensure_postgres_schema()
    import psycopg

    ticker = ticker.upper()
    with psycopg.connect(database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT ts FROM snapshots WHERE ticker = %s", (ticker,))
            return {str(row[0]) for row in cur.fetchall()}


def import_export_timestamp(
    ticker: str,
    ts: str,
    *,
    export_dir: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> ImportResult:
    """Import one export timestamp into PostgreSQL."""
    from gex_core.db import use_postgres
    from gex_core.snapshot_export import write_snapshot_export

    ticker = ticker.upper()
    if not use_postgres():
        return ImportResult(ticker=ticker, ts=ts, status="error", error="DATABASE_URL not set")

    if not force and ts in list_postgres_timestamps(ticker):
        return ImportResult(ticker=ticker, ts=ts, status="skipped_existing", skipped=True)

    payload = load_export_snapshot(ticker, ts, export_dir=export_dir)
    if payload is None:
        return ImportResult(ticker=ticker, ts=ts, status="missing_files", skipped=True)

    if dry_run:
        return ImportResult(ticker=ticker, ts=ts, status="dry_run", written=False)

    try:
        write_snapshot_export(
            ticker,
            gex_by_strike=payload["gex_by_strike"],
            cumulative_gex=payload["cumulative_gex"],
            gex_by_expiration=payload["gex_by_expiration"],
            surface_data=payload["surface_data"],
            greek_exposure_df=payload["greek_exposure_df"],
            summary=payload["summary"],
            export_dir=export_dir or EXPORT_DIR,
            timestamp=ts,
            force=force,
        )
    except Exception as exc:
        logger.exception("Failed to import %s %s", ticker, ts)
        return ImportResult(ticker=ticker, ts=ts, status="error", error=str(exc))

    return ImportResult(ticker=ticker, ts=ts, status="imported", written=True)


def import_ticker_exports(
    ticker: str,
    *,
    export_dir: Path | None = None,
    timestamps: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> list[ImportResult]:
    """Bulk-import CSV export sets for one ticker."""
    export_dir = export_dir or EXPORT_DIR
    ticker = ticker.upper()
    ts_list = timestamps or scan_export_timestamps(ticker, export_dir)
    existing = list_postgres_timestamps(ticker) if skip_existing and not force else set()
    results: list[ImportResult] = []
    for ts in ts_list:
        if skip_existing and not force and ts in existing:
            results.append(ImportResult(ticker=ticker, ts=ts, status="skipped_existing", skipped=True))
            continue
        results.append(
            import_export_timestamp(
                ticker,
                ts,
                export_dir=export_dir,
                force=force,
                dry_run=dry_run,
            )
        )
    return results


def summarize_import_results(results: list[ImportResult]) -> dict[str, int]:
    counts = {"imported": 0, "skipped": 0, "errors": 0, "dry_run": 0}
    for result in results:
        if result.status == "imported":
            counts["imported"] += 1
        elif result.skipped or result.status == "skipped_existing":
            counts["skipped"] += 1
        elif result.status == "dry_run":
            counts["dry_run"] += 1
        elif result.status == "error":
            counts["errors"] += 1
    return counts
