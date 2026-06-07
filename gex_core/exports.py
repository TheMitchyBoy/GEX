"""Shared utilities for scanning and loading GEX CSV exports.

Each UW refresh writes a matched set of files sharing a timestamp suffix::

    {TICKER}_gex_by_strike_{ts}.csv
    {TICKER}_cumulative_gex_{ts}.csv
    {TICKER}_gex_by_expiration_{ts}.csv
    {TICKER}_gex_surface_{ts}.csv      (when surface data is present)
    {TICKER}_summary_{ts}.json

``{ts}`` is ``YYYY-MM-DD_HHMMSS``. Dashboards discover history by globbing these
patterns; an optional SQLite index (``gex_core.storage``) accelerates lookups.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Resolve relative to repo root so gunicorn/docker cwd does not break history.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_export_dir() -> Path:
    import os

    raw = os.environ.get("GEX_EXPORT_DIR", "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else _REPO_ROOT / path
    return _REPO_ROOT / "data" / "exports"


EXPORT_DIR = _resolve_export_dir()


def refresh_export_dir() -> Path:
    """Re-read ``GEX_EXPORT_DIR`` after ``configure_data_paths()`` runs."""
    global EXPORT_DIR
    EXPORT_DIR = _resolve_export_dir()
    return EXPORT_DIR

TIMESTAMP_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9]+)_(?P<kind>gex_by_strike|gex_by_expiration|gex_surface|cumulative_gex)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{6})\.csv$"
)
SUMMARY_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9]+)_summary_(?P<ts>\d{4}-\d{2}-\d{2}_\d{6})\.json$"
)


def parse_timestamp(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, "%Y-%m-%d_%H%M%S")


def paths_for_export_timestamp(ticker: str, ts: str, export_dir: Path | None = None) -> dict[str, Path]:
    """Resolve export paths for one timestamp without scanning the export directory."""
    export_dir = export_dir or EXPORT_DIR
    ticker = ticker.upper()
    kinds: dict[str, Path] = {}
    for kind in ("gex_by_strike", "cumulative_gex", "gex_by_expiration", "gex_surface"):
        path = export_dir / f"{ticker}_{kind}_{ts}.csv"
        if path.exists():
            kinds[kind] = path
    summary = export_dir / f"{ticker}_summary_{ts}.json"
    if summary.exists():
        kinds["summary"] = summary
    return kinds


def scan_export_timestamps(ticker: str, export_dir: Path | None = None) -> list[str]:
    """List export timestamps from strike filenames (one glob pass)."""
    export_dir = export_dir or EXPORT_DIR
    timestamps: set[str] = set()
    for path in export_dir.glob(f"{ticker.upper()}_gex_by_strike_*_*.csv"):
        match = TIMESTAMP_RE.match(path.name)
        if match:
            timestamps.add(match.group("ts"))
    return sorted(timestamps)


def list_export_timestamps(ticker: str, export_dir: Path | None = None) -> list[str]:
    """List export timestamps that have a strike CSV on disk.

    Uses the SQLite index when it matches on-disk files; otherwise trusts the
    directory scan (handles stale index rows after redeploys).
    """
    export_dir = export_dir or EXPORT_DIR
    on_disk = scan_export_timestamps(ticker, export_dir)
    if export_dir.resolve() != EXPORT_DIR.resolve():
        return on_disk
    try:
        from gex_core.storage import list_indexed_timestamps, sync_ticker_exports

        sync_ticker_exports(ticker, export_dir)
        indexed = list_indexed_timestamps(ticker)
        if indexed and len(indexed) == len(on_disk):
            return indexed
        if indexed and on_disk:
            disk_set = set(on_disk)
            verified = [ts for ts in indexed if ts in disk_set]
            if len(verified) == len(on_disk):
                return verified
    except Exception as exc:
        logger.debug("Indexed export timestamps unavailable for %s: %s", ticker, exc)
    return on_disk


def filter_export_timestamps(
    timestamps: list[str],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    max_timestamps: int | None = None,
) -> list[str]:
    """Trim a sorted timestamp list to a time window and optional cap."""
    if not timestamps:
        return []
    filtered = timestamps
    if since is not None:
        filtered = [ts for ts in filtered if parse_timestamp(ts) >= since]
    if until is not None:
        filtered = [ts for ts in filtered if parse_timestamp(ts) <= until]
    if max_timestamps is not None and max_timestamps > 0 and len(filtered) > max_timestamps:
        filtered = filtered[-max_timestamps:]
    return filtered


def find_exports_for_ticker(
    ticker: str,
    export_dir: Path | None = None,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    max_timestamps: int | None = None,
) -> dict[str, dict[str, Path]]:
    """Return {timestamp_str: {kind: Path}} for a ticker.

    When ``since`` / ``max_timestamps`` are set, only resolves paths for the
    requested window instead of globbing every export file.
    """
    export_dir = export_dir or EXPORT_DIR
    if since is not None or until is not None or max_timestamps is not None:
        timestamps = filter_export_timestamps(
            list_export_timestamps(ticker, export_dir),
            since=since,
            until=until,
            max_timestamps=max_timestamps,
        )
        records: dict[str, dict[str, Path]] = {}
        for ts in timestamps:
            kinds = paths_for_export_timestamp(ticker, ts, export_dir)
            if kinds:
                records[ts] = kinds
        return records

    timestamps = list_export_timestamps(ticker, export_dir)
    if timestamps:
        records = {}
        for ts in timestamps:
            kinds = paths_for_export_timestamp(ticker, ts, export_dir)
            if kinds:
                records[ts] = kinds
        if records:
            return records

    records = {}
    for path in export_dir.glob(f"{ticker.upper()}_*_*_*.csv"):
        match = TIMESTAMP_RE.match(path.name)
        if not match:
            continue
        ts = match.group("ts")
        kind = match.group("kind")
        records.setdefault(ts, {})[kind] = path
    for path in export_dir.glob(f"{ticker.upper()}_summary_*.json"):
        match = SUMMARY_RE.match(path.name)
        if not match:
            continue
        records.setdefault(match.group("ts"), {})["summary"] = path
    return records


def load_strike_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    idx = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0.0)
    return pd.Series(vals.values, index=idx.values, name="gex_bn")


def load_expiration_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    idx = df.iloc[:, 0]
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0.0)
    return pd.Series(vals.values, index=idx.values, name="gex_bn")


def load_cumulative_series(path: Path) -> pd.Series:
    return load_strike_series(path)


def load_surface_df(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size < 2:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    if "GEX" in df.columns:
        df["GEX"] = pd.to_numeric(df["GEX"], errors="coerce")
    return df
