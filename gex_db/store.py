from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

CSV_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9]+)_(?P<kind>gex_by_strike|gex_by_expiration|gex_surface|cumulative_gex)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{6})\.csv$"
)

DEFAULT_DB_PATH = Path(os.environ.get("GEX_DB_PATH", "data/gex.db"))


def get_db_path() -> Path:
    return DEFAULT_DB_PATH


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gex_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                ts TEXT NOT NULL,
                created_at TEXT NOT NULL,
                strike_json TEXT NOT NULL,
                expiration_json TEXT NOT NULL,
                cumulative_json TEXT NOT NULL,
                surface_json TEXT,
                summary_json TEXT,
                UNIQUE (ticker, ts)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_gex_snapshots_ticker_ts ON gex_snapshots (ticker, ts)"
        )


def _series_to_records(series: pd.Series) -> list[list[float]]:
    records: list[list[float]] = []
    for index, value in series.items():
        try:
            strike = float(index)
            gex = float(value)
        except (TypeError, ValueError):
            continue
        records.append([strike, gex])
    return records


def _records_to_series(records: list[list[float]], value_name: str = "gex") -> pd.Series:
    if not records:
        return pd.Series(dtype=float)
    strikes = [float(row[0]) for row in records]
    values = [float(row[1]) for row in records]
    return pd.Series(values, index=strikes, name=value_name)


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.copy()
    if "expiration" in out.columns:
        out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce").dt.strftime("%Y-%m-%d")
    return json.loads(out.to_json(orient="records"))


def _records_to_surface_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["expiration", "strike", "GEX"])
    df = pd.DataFrame(records)
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    if "GEX" in df.columns:
        df["GEX"] = pd.to_numeric(df["GEX"], errors="coerce")
    return df


def save_snapshot(
    ticker: str,
    ts: str,
    gex_by_strike: pd.Series,
    cumulative_gex: pd.Series,
    gex_by_expiration: pd.Series,
    surface_data: pd.DataFrame,
    summary: dict | None = None,
    db_path: Path | None = None,
) -> None:
    init_db(db_path)
    ticker = ticker.upper()
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload = (
        ticker,
        ts,
        created_at,
        json.dumps(_series_to_records(gex_by_strike)),
        json.dumps(_series_to_records(gex_by_expiration)),
        json.dumps(_series_to_records(cumulative_gex)),
        json.dumps(_df_to_records(surface_data)),
        json.dumps(summary) if summary is not None else None,
    )
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO gex_snapshots (
                ticker, ts, created_at, strike_json, expiration_json,
                cumulative_json, surface_json, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, ts) DO UPDATE SET
                created_at = excluded.created_at,
                strike_json = excluded.strike_json,
                expiration_json = excluded.expiration_json,
                cumulative_json = excluded.cumulative_json,
                surface_json = excluded.surface_json,
                summary_json = excluded.summary_json
            """,
            payload,
        )


def list_timestamps(ticker: str, db_path: Path | None = None) -> list[str]:
    init_db(db_path)
    ticker = ticker.upper()
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT ts FROM gex_snapshots WHERE ticker = ? ORDER BY ts ASC",
            (ticker,),
        ).fetchall()
    return [row["ts"] for row in rows]


def list_tickers(db_path: Path | None = None) -> list[str]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM gex_snapshots ORDER BY ticker ASC"
        ).fetchall()
    return [row["ticker"] for row in rows]


def get_latest_ts(ticker: str, db_path: Path | None = None) -> str | None:
    timestamps = list_timestamps(ticker, db_path=db_path)
    return timestamps[-1] if timestamps else None


def get_snapshot(ticker: str, ts: str, db_path: Path | None = None) -> dict | None:
    init_db(db_path)
    ticker = ticker.upper()
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM gex_snapshots WHERE ticker = ? AND ts = ?",
            (ticker, ts),
        ).fetchone()
    if row is None:
        return None

    strike = _records_to_series(json.loads(row["strike_json"]), "gex_bn_per_pct")
    expiration = _records_to_series(json.loads(row["expiration_json"]), "gex_bn_per_pct")
    cumulative = _records_to_series(json.loads(row["cumulative_json"]), "cumulative_gex_bn_per_pct")
    surface_records = json.loads(row["surface_json"] or "[]")
    surface_df = _records_to_surface_df(surface_records)
    summary = json.loads(row["summary_json"]) if row["summary_json"] else None

    return {
        "ticker": ticker,
        "ts": row["ts"],
        "created_at": row["created_at"],
        "strike": strike,
        "expiration": expiration,
        "cumulative": cumulative,
        "surface_df": surface_df,
        "summary": summary,
    }


def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d_%H%M%S")


def is_snapshot_stale(ticker: str, max_age_minutes: int = 10, db_path: Path | None = None) -> bool:
    latest = get_latest_ts(ticker, db_path=db_path)
    if latest is None:
        return True
    age = datetime.now() - parse_ts(latest)
    return age > timedelta(minutes=max_age_minutes)


def _load_csv_snapshot_files(export_dir: Path, ticker: str) -> dict[str, dict[str, Path]]:
    snapshots: dict[str, dict[str, Path]] = {}
    for file in export_dir.glob(f"{ticker}_*.csv"):
        match = CSV_RE.match(file.name)
        if not match:
            continue
        ts = match.group("ts")
        kind = match.group("kind")
        snapshots.setdefault(ts, {})[kind] = file

    return {
        ts: files
        for ts, files in snapshots.items()
        if {"gex_by_strike", "gex_by_expiration", "cumulative_gex"}.issubset(files)
    }


def _import_csv_snapshot(ticker: str, ts: str, files: dict[str, Path], db_path: Path | None = None) -> bool:
    try:
        strike_df = pd.read_csv(files["gex_by_strike"])
        exp_df = pd.read_csv(files["gex_by_expiration"])
        cum_df = pd.read_csv(files["cumulative_gex"])
        surface_df = pd.read_csv(files["gex_surface"]) if "gex_surface" in files else pd.DataFrame()

        strike = pd.Series(
            pd.to_numeric(strike_df.iloc[:, 1], errors="coerce").fillna(0.0).values,
            index=pd.to_numeric(strike_df.iloc[:, 0], errors="coerce"),
        )
        expiration = pd.Series(
            pd.to_numeric(exp_df.iloc[:, 1], errors="coerce").fillna(0.0).values,
            index=pd.to_datetime(exp_df.iloc[:, 0], errors="coerce"),
        )
        cumulative = pd.Series(
            pd.to_numeric(cum_df.iloc[:, 1], errors="coerce").fillna(0.0).values,
            index=pd.to_numeric(cum_df.iloc[:, 0], errors="coerce"),
        )
        save_snapshot(
            ticker=ticker,
            ts=ts,
            gex_by_strike=strike,
            cumulative_gex=cumulative,
            gex_by_expiration=expiration,
            surface_data=surface_df,
            db_path=db_path,
        )
        return True
    except Exception:
        return False


def import_csv_exports(
    export_dir: Path | None = None,
    ticker: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Import CSV exports into the database. Returns number of snapshots imported."""
    export_dir = export_dir or Path("data/exports")
    if not export_dir.exists():
        return 0

    init_db(db_path)
    imported = 0
    tickers = [ticker.upper()] if ticker else sorted({m.group("ticker") for m in (CSV_RE.match(f.name) for f in export_dir.glob("*.csv")) if m})

    for symbol in tickers:
        snapshot_files = _load_csv_snapshot_files(export_dir, symbol)
        existing = set(list_timestamps(symbol, db_path=db_path))
        for ts, files in snapshot_files.items():
            if ts in existing:
                continue
            if _import_csv_snapshot(symbol, ts, files, db_path=db_path):
                imported += 1
    return imported
