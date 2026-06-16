"""SQLite index over file-based GEX exports for fast history lookups."""

from __future__ import annotations

import logging
import os
import time
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

_SYNC_CACHE: dict[tuple[str, str], float] = {}

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = _REPO_ROOT / "data" / "gex_index.db"


def db_path() -> Path:
    raw = os.environ.get("GEX_INDEX_DB", "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else _REPO_ROOT / path
    return DEFAULT_DB_PATH


def _connect(path: Path | None = None):
    from gex_core.db import get_connection

    return get_connection(group="index", sqlite_path=db_path(), sqlite_path_override=path)


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
    except Exception as exc:
        logger.warning("latest_timestamp failed: %s", exc)
    from gex_core.db import use_postgres

    if use_postgres() and path is None:
        return None
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
    except Exception as exc:
        logger.warning("SQLite list_indexed_timestamps failed: %s", exc)
    return []


def list_indexed_dates(ticker: str, path: Path | None = None) -> list[str]:
    """Distinct trading days in the index (no CSV glob)."""
    ticker = ticker.upper()
    try:
        with _connect(path) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT COALESCE(market_date, substr(ts, 1, 10)) AS day
                FROM snapshots
                WHERE ticker = ?
                ORDER BY day ASC
                """,
                (ticker,),
            ).fetchall()
        return [str(r["day"]) for r in rows if r["day"]]
    except Exception as exc:
        logger.warning("SQLite list_indexed_dates failed: %s", exc)
        return []


def list_indexed_timestamps_for_date(
    ticker: str,
    market_date: str,
    path: Path | None = None,
) -> list[str]:
    """All slice timestamps for one day from the index."""
    ticker = ticker.upper()
    market_date = market_date[:10]
    try:
        with _connect(path) as conn:
            rows = conn.execute(
                """
                SELECT ts FROM snapshots
                WHERE ticker = ?
                  AND COALESCE(market_date, substr(ts, 1, 10)) = ?
                ORDER BY ts ASC
                """,
                (ticker, market_date),
            ).fetchall()
        return [str(r["ts"]) for r in rows]
    except Exception as exc:
        logger.warning("SQLite list_indexed_timestamps_for_date failed: %s", exc)
        return []


def list_indexed_timestamps_before_date(
    ticker: str,
    market_date: str,
    path: Path | None = None,
) -> list[str]:
    """Indexed timestamps strictly before a calendar day (historical fast path)."""
    ticker = ticker.upper()
    market_date = market_date[:10]
    try:
        with _connect(path) as conn:
            rows = conn.execute(
                """
                SELECT ts FROM snapshots
                WHERE ticker = ?
                  AND COALESCE(market_date, substr(ts, 1, 10)) < ?
                ORDER BY ts ASC
                """,
                (ticker, market_date),
            ).fetchall()
        return [str(r["ts"]) for r in rows]
    except Exception as exc:
        logger.warning("SQLite list_indexed_timestamps_before_date failed: %s", exc)
        return []


def strike_export_path(ticker: str, ts: str, export_dir: Path | None = None) -> Path:
    export_dir = export_dir or EXPORT_DIR
    return export_dir / f"{ticker.upper()}_gex_by_strike_{ts}.csv"


def count_strike_exports_on_disk(ticker: str, export_dir: Path | None = None) -> int:
    """Count strike CSV snapshots present on disk (ignores the SQLite index)."""
    return len(scan_export_timestamps(ticker, export_dir or EXPORT_DIR))


def prune_stale_index_entries(
    ticker: str,
    export_dir: Path | None = None,
    path: Path | None = None,
) -> int:
    """Remove index rows whose strike CSV no longer exists on disk."""
    export_dir = export_dir or EXPORT_DIR
    ticker = ticker.upper()
    on_disk = set(scan_export_timestamps(ticker, export_dir))
    if not on_disk:
        return 0
    indexed = list_indexed_timestamps(ticker, path)
    stale = [ts for ts in indexed if ts not in on_disk]
    if not stale:
        return 0
    try:
        with _connect(path) as conn:
            conn.executemany(
                "DELETE FROM snapshots WHERE ticker = ? AND ts = ?",
                [(ticker, ts) for ts in stale],
            )
            conn.commit()
    except Exception as exc:
        logger.warning("SQLite prune_stale_index_entries failed: %s", exc)
        return 0
    logger.info(
        "Pruned %d stale %s index entries (strike CSV missing under %s)",
        len(stale),
        ticker,
        export_dir,
    )
    return len(stale)


def _sync_cache_ttl_sec() -> float:
    try:
        return max(15.0, float(os.environ.get("GEX_INDEX_SYNC_TTL_SEC", "120")))
    except (TypeError, ValueError):
        return 120.0


def clear_sync_cache(ticker: str | None = None, export_dir: Path | None = None) -> None:
    """Drop sync throttle entries (e.g. after a new export lands)."""
    if ticker is None and export_dir is None:
        _SYNC_CACHE.clear()
        return
    export_dir = export_dir or EXPORT_DIR
    key = (ticker.upper() if ticker else "", str(export_dir.resolve()))
    _SYNC_CACHE.pop(key, None)


def sync_ticker_exports(
    ticker: str,
    export_dir: Path | None = None,
    path: Path | None = None,
    *,
    force: bool = False,
) -> int:
    """Reconcile SQLite index with on-disk strike exports."""
    import json

    export_dir = export_dir or EXPORT_DIR
    ticker = ticker.upper()
    cache_key = (ticker, str(export_dir.resolve()))
    if not force:
        last_run = _SYNC_CACHE.get(cache_key)
        if last_run is not None and (time.monotonic() - last_run) < _sync_cache_ttl_sec():
            return 0
    prune_stale_index_entries(ticker, export_dir, path)
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
    _SYNC_CACHE[cache_key] = time.monotonic()
    return added


def export_age_minutes(ticker: str, export_dir: Path | None = None) -> float | None:
    latest = latest_timestamp(ticker, export_dir=export_dir)
    if not latest:
        return None
    age = datetime.now() - parse_timestamp(latest)
    return age.total_seconds() / 60.0


def fetch_index_spot_series(
    ticker: str,
    *,
    days: int = 90,
    interval_minutes: int = 10,
    max_points: int | None = 500,
    export_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Lightweight spot/GEX series from the SQLite index for charts."""
    from datetime import timedelta

    ticker = ticker.upper()
    export_dir = export_dir or EXPORT_DIR
    sync_ticker_exports(ticker, export_dir)
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, spot, total_gex, regime
                FROM snapshots
                WHERE ticker = ?
                ORDER BY ts ASC
                """,
                (ticker,),
            ).fetchall()
    except Exception as exc:
        logger.warning("fetch_index_spot_series failed: %s", exc)
        return []

    if not rows:
        return []

    latest = parse_timestamp(str(rows[-1]["ts"]))
    cutoff = latest - timedelta(days=max(1, days))
    series: list[dict[str, Any]] = []
    last_bucket: datetime | None = None
    for row in rows:
        ts = str(row["ts"])
        spot = row["spot"]
        if spot is None:
            continue
        ts_dt = parse_timestamp(ts)
        if ts_dt < cutoff:
            continue
        if interval_minutes > 1:
            bucket = ts_dt.replace(second=0, microsecond=0) - timedelta(
                minutes=ts_dt.minute % interval_minutes
            )
            if last_bucket is not None and bucket <= last_bucket:
                series[-1] = {
                    "ts": ts,
                    "ts_label": ts_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "spot": float(spot),
                    "total_gex": float(row["total_gex"]) if row["total_gex"] is not None else None,
                    "regime": row["regime"],
                }
                continue
            last_bucket = bucket
        series.append(
            {
                "ts": ts,
                "ts_label": ts_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "spot": float(spot),
                "total_gex": float(row["total_gex"]) if row["total_gex"] is not None else None,
                "regime": row["regime"],
            }
        )

    if max_points and len(series) > max_points:
        step = max(1, len(series) // max_points)
        series = series[::step]
        if series[-1]["ts"] != rows[-1]["ts"]:
            last_row = rows[-1]
            if last_row["spot"] is not None:
                ts = str(last_row["ts"])
                ts_dt = parse_timestamp(ts)
                series.append(
                    {
                        "ts": ts,
                        "ts_label": ts_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "spot": float(last_row["spot"]),
                        "total_gex": float(last_row["total_gex"])
                        if last_row["total_gex"] is not None
                        else None,
                        "regime": last_row["regime"],
                    }
                )
    return series
