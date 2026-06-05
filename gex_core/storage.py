"""SQLite index over file-based GEX exports for fast history lookups."""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gex_core.exports import (
    EXPORT_DIR,
    paths_for_export_timestamp,
    parse_timestamp,
    scan_export_timestamps,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = _REPO_ROOT / "data" / "gex_index.db"


def db_path() -> Path:
    raw = Path(os.environ.get("GEX_INDEX_DB", str(DEFAULT_DB_PATH)))
    if not raw.is_absolute():
        return _REPO_ROOT / raw
    return raw


def _connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            ticker TEXT NOT NULL,
            ts TEXT NOT NULL,
            market_date TEXT,
            spot REAL,
            total_gex REAL,
            regime TEXT,
            summary_path TEXT,
            strike_path TEXT,
            indexed_at TEXT,
            PRIMARY KEY (ticker, ts)
        );
        CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_ts
            ON snapshots (ticker, ts DESC);
        """
    )
    return conn


def upsert_snapshot(
    ticker: str,
    ts: str,
    *,
    market_date: str | None = None,
    spot: float | None = None,
    total_gex: float | None = None,
    regime: str | None = None,
    summary_path: str | None = None,
    strike_path: str | None = None,
    path: Path | None = None,
) -> None:
    ticker = ticker.upper()
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO snapshots (
                ticker, ts, market_date, spot, total_gex, regime,
                summary_path, strike_path, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, ts) DO UPDATE SET
                market_date=excluded.market_date,
                spot=excluded.spot,
                total_gex=excluded.total_gex,
                regime=excluded.regime,
                summary_path=excluded.summary_path,
                strike_path=excluded.strike_path,
                indexed_at=excluded.indexed_at
            """,
            (
                ticker,
                ts,
                market_date,
                spot,
                total_gex,
                regime,
                summary_path,
                strike_path,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def latest_timestamp(ticker: str, export_dir: Path | None = None, path: Path | None = None) -> str | None:
    ticker = ticker.upper()
    try:
        with _connect(path) as conn:
            row = conn.execute(
                "SELECT ts FROM snapshots WHERE ticker = ? ORDER BY ts DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        if row:
            return str(row["ts"])
    except sqlite3.Error as exc:
        logger.warning("SQLite latest_timestamp failed: %s", exc)
    timestamps = scan_export_timestamps(ticker, export_dir or EXPORT_DIR)
    return timestamps[-1] if timestamps else None


def list_indexed_timestamps(ticker: str, path: Path | None = None) -> list[str]:
    ticker = ticker.upper()
    try:
        with _connect(path) as conn:
            rows = conn.execute(
                "SELECT ts FROM snapshots WHERE ticker = ? ORDER BY ts ASC",
                (ticker,),
            ).fetchall()
        if rows:
            return [str(r["ts"]) for r in rows]
    except sqlite3.Error as exc:
        logger.warning("SQLite list_indexed_timestamps failed: %s", exc)
    return []


def sync_ticker_exports(ticker: str, export_dir: Path | None = None, path: Path | None = None) -> int:
    """Index any export timestamps missing from the SQLite catalog."""
    import json

    export_dir = export_dir or EXPORT_DIR
    ticker = ticker.upper()
    existing = set(list_indexed_timestamps(ticker, path))
    timestamps = [ts for ts in scan_export_timestamps(ticker, export_dir) if ts not in existing]
    added = 0
    for ts in timestamps:
        kinds = paths_for_export_timestamp(ticker, ts, export_dir)
        if "gex_by_strike" not in kinds:
            continue
        summary_path = kinds.get("summary")
        spot = None
        total_gex = None
        regime = None
        market_date = None
        if summary_path and summary_path.exists():
            try:
                with summary_path.open(encoding="utf-8") as f:
                    summary = json.load(f)
                spot = summary.get("spot") or summary.get("spot_price")
                total_gex = summary.get("total_gex_bn_per_pct")
                regime = summary.get("net_gamma_regime")
                market_date = summary.get("market_date")
                if market_date is None and ts:
                    market_date = ts.split("_", 1)[0]
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("Could not read summary for %s %s: %s", ticker, ts, exc)
        upsert_snapshot(
            ticker,
            ts,
            market_date=market_date,
            spot=float(spot) if spot is not None else None,
            total_gex=float(total_gex) if total_gex is not None else None,
            regime=str(regime) if regime else None,
            summary_path=str(summary_path) if summary_path else None,
            strike_path=str(kinds["gex_by_strike"]),
            path=path,
        )
        added += 1
    return added


def export_age_minutes(ticker: str, export_dir: Path | None = None) -> float | None:
    latest = latest_timestamp(ticker, export_dir=export_dir)
    if not latest:
        return None
    age = datetime.now() - parse_timestamp(latest)
    return age.total_seconds() / 60.0
