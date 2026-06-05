"""Shared utilities for scanning and loading GEX CSV exports."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

# Resolve relative to repo root so gunicorn/docker cwd does not break history.
_REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = _REPO_ROOT / "data" / "exports"

TIMESTAMP_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9]+)_(?P<kind>gex_by_strike|gex_by_expiration|gex_surface|cumulative_gex)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{6})\.csv$"
)
SUMMARY_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9]+)_summary_(?P<ts>\d{4}-\d{2}-\d{2}_\d{6})\.json$"
)


def parse_timestamp(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, "%Y-%m-%d_%H%M%S")


def find_exports_for_ticker(ticker: str, export_dir: Path | None = None) -> dict[str, dict[str, Path]]:
    """Return {timestamp_str: {kind: Path}} for a ticker."""
    export_dir = export_dir or EXPORT_DIR
    records: dict[str, dict[str, Path]] = {}
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
