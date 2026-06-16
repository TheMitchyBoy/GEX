"""Migrate legacy SQLite state and export index metadata into PostgreSQL."""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from gex_core.db import ensure_postgres_schema, use_postgres

logger = logging.getLogger(__name__)

_BATCH_SIZE = 500


@dataclass
class MigrationStats:
    snapshots: int = 0
    trades: int = 0
    decisions: int = 0
    trader_state: int = 0
    llm_predictions: int = 0
    daily_insights: int = 0
    export_sync_added: int = 0
    skipped: bool = False
    reason: str = ""

    def total_rows(self) -> int:
        return (
            self.snapshots
            + self.trades
            + self.decisions
            + self.trader_state
            + self.llm_predictions
            + self.daily_insights
        )


def index_db_path() -> Path:
    from gex_core.storage import db_path

    return db_path()


def journal_db_path() -> Path:
    from gex_core.trading.journal import db_path

    return db_path()


def _sqlite_table_exists(db_path: Path, table: str) -> bool:
    if not db_path.exists():
        return False
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
    return row is not None


def _read_sqlite_rows(db_path: Path, table: str) -> list[dict[str, Any]]:
    if not _sqlite_table_exists(db_path, table):
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    return [dict(row) for row in rows]


def _sqlite_row_count(db_path: Path, table: str) -> int:
    if not _sqlite_table_exists(db_path, table):
        return 0
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return int(row[0]) if row else 0


def _postgres_row_count(table: str) -> int:
    import psycopg

    from gex_core.db import database_url

    with psycopg.connect(database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            row = cur.fetchone()
    return int(row[0]) if row else 0


def sqlite_has_data() -> bool:
    from gex_core.runtime_mode import is_processor_mode

    index_db = index_db_path()
    if is_processor_mode():
        return _sqlite_row_count(index_db, "snapshots") > 0
    journal_db = journal_db_path()
    counts = (
        _sqlite_row_count(index_db, "snapshots")
        + _sqlite_row_count(journal_db, "trades")
        + _sqlite_row_count(journal_db, "decisions")
        + _sqlite_row_count(journal_db, "trader_state")
        + _sqlite_row_count(journal_db, "llm_predictions")
        + _sqlite_row_count(journal_db, "daily_insights")
    )
    return counts > 0


def postgres_is_empty() -> bool:
    from gex_core.runtime_mode import is_processor_mode

    if is_processor_mode():
        return _postgres_row_count("snapshots") == 0
    counts = (
        _postgres_row_count("snapshots")
        + _postgres_row_count("trades")
        + _postgres_row_count("decisions")
        + _postgres_row_count("trader_state")
        + _postgres_row_count("llm_predictions")
        + _postgres_row_count("daily_insights")
    )
    return counts == 0


def should_auto_migrate() -> bool:
    if not use_postgres():
        return False
    flag = os.environ.get("GEX_MIGRATE_SQLITE", "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    if not sqlite_has_data():
        return False
    if postgres_is_empty():
        return True
    try:
        return _sqlite_row_count(index_db_path(), "snapshots") > _postgres_row_count("snapshots")
    except Exception as exc:
        logger.warning("Auto-migrate snapshot count check failed: %s", exc)
        return False


def _chunks(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _upsert_batches(
    conn: Any,
    *,
    table: str,
    columns: list[str],
    conflict_cols: list[str],
    rows: list[dict[str, Any]],
    update_cols: list[str] | None = None,
) -> int:
    if not rows:
        return 0
    update_cols = update_cols or [col for col in columns if col not in conflict_cols]
    set_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_cols)
    conflict = ", ".join(conflict_cols)
    placeholders = ", ".join(["%s"] * len(columns))
    col_list = ", ".join(columns)
    sql = f"""
        INSERT INTO {table} ({col_list})
        VALUES ({placeholders})
        ON CONFLICT ({conflict}) DO UPDATE SET {set_clause}
    """
    migrated = 0
    with conn.cursor() as cur:
        for batch in _chunks(rows, _BATCH_SIZE):
            params = [tuple(row.get(col) for col in columns) for row in batch]
            cur.executemany(sql, params)
            migrated += len(batch)
    return migrated


def _insert_batches_with_ids(
    conn: Any,
    *,
    table: str,
    columns: list[str],
    rows: list[dict[str, Any]],
) -> int:
    if not rows:
        return 0
    placeholders = ", ".join(["%s"] * len(columns))
    col_list = ", ".join(columns)
    sql = f"""
        INSERT INTO {table} ({col_list})
        VALUES ({placeholders})
        ON CONFLICT (id) DO UPDATE SET
        {", ".join(f"{col} = EXCLUDED.{col}" for col in columns if col != "id")}
    """
    migrated = 0
    with conn.cursor() as cur:
        for batch in _chunks(rows, _BATCH_SIZE):
            params = [tuple(row.get(col) for col in columns) for row in batch]
            cur.executemany(sql, params)
            migrated += len(batch)
    return migrated


def _reset_sequence(conn: Any, table: str, *, column: str = "id") -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT setval(
                pg_get_serial_sequence('{table}', '{column}'),
                COALESCE((SELECT MAX({column}) FROM {table}), 1),
                (SELECT MAX({column}) IS NOT NULL FROM {table})
            )
            """
        )


def _tickers_from_exports(export_dir: Path) -> list[str]:
    tickers: set[str] = set()
    if not export_dir.exists():
        return []
    for path in export_dir.glob("*_gex_by_strike_*.csv"):
        tickers.add(path.name.split("_gex_by_strike_", 1)[0].upper())
    return sorted(tickers)


def migrate_sqlite_to_postgres(*, force: bool = False, sync_exports: bool = True) -> MigrationStats:
    """Copy SQLite tables and reconcile export index metadata into PostgreSQL."""
    from gex_core.runtime_mode import is_processor_mode

    _ = force  # Reserved for CLI; upserts are always idempotent.
    stats = MigrationStats()
    if not use_postgres():
        stats.skipped = True
        stats.reason = "DATABASE_URL is not set"
        return stats

    ensure_postgres_schema(processor_only=is_processor_mode())

    has_sqlite = sqlite_has_data()
    if not has_sqlite and not sync_exports:
        stats.skipped = True
        stats.reason = "No SQLite data and export sync disabled"
        return stats

    import psycopg
    from gex_core.db import database_url

    processor_only = is_processor_mode()
    index_db = index_db_path()
    journal_db = journal_db_path()
    snapshot_rows = _read_sqlite_rows(index_db, "snapshots") if has_sqlite else []
    trade_rows: list[dict] = []
    decision_rows: list[dict] = []
    state_rows: list[dict] = []
    prediction_rows: list[dict] = []
    insight_rows: list[dict] = []
    if has_sqlite and not processor_only:
        trade_rows = _read_sqlite_rows(journal_db, "trades")
        decision_rows = _read_sqlite_rows(journal_db, "decisions")
        state_rows = _read_sqlite_rows(journal_db, "trader_state")
        prediction_rows = _read_sqlite_rows(journal_db, "llm_predictions")
        insight_rows = _read_sqlite_rows(journal_db, "daily_insights")

    if has_sqlite:
        with psycopg.connect(database_url()) as conn:
            stats.snapshots = _upsert_batches(
                conn,
                table="snapshots",
                columns=[
                    "ticker",
                    "ts",
                    "market_date",
                    "spot",
                    "total_gex",
                    "regime",
                    "summary_path",
                    "strike_path",
                    "indexed_at",
                ],
                conflict_cols=["ticker", "ts"],
                rows=snapshot_rows,
            )
            stats.trades = _insert_batches_with_ids(
                conn,
                table="trades",
                columns=[
                    "id",
                    "ticker",
                    "status",
                    "option_type",
                    "strike",
                    "qty",
                    "entry_ts",
                    "exit_ts",
                    "entry_spot",
                    "exit_spot",
                    "entry_premium",
                    "exit_premium",
                    "pnl_pct",
                    "pnl_usd",
                    "exit_reason",
                    "signal_type",
                    "signal_strike",
                    "signal_gamma",
                    "gamma_delta",
                    "ai_confidence",
                    "ai_reason",
                    "meta_json",
                ],
                rows=trade_rows,
            )
            stats.decisions = _insert_batches_with_ids(
                conn,
                table="decisions",
                columns=["id", "ts", "ticker", "action", "payload_json", "ai_verdict", "ai_notes"],
                rows=decision_rows,
            )
            stats.trader_state = _upsert_batches(
                conn,
                table="trader_state",
                columns=["key", "value", "updated_at"],
                conflict_cols=["key"],
                rows=state_rows,
            )
            stats.llm_predictions = _insert_batches_with_ids(
                conn,
                table="llm_predictions",
                columns=[
                    "id",
                    "ticker",
                    "source",
                    "snapshot_ts",
                    "market_date",
                    "created_at",
                    "resolved_at",
                    "payload_json",
                    "actual_json",
                    "outcome_json",
                ],
                rows=prediction_rows,
            )
            stats.daily_insights = _upsert_batches(
                conn,
                table="daily_insights",
                columns=["ticker", "market_date", "kind", "payload_json", "created_at", "updated_at"],
                conflict_cols=["ticker", "market_date", "kind"],
                rows=insight_rows,
            )

            if trade_rows:
                _reset_sequence(conn, "trades")
            if decision_rows:
                _reset_sequence(conn, "decisions")
            if prediction_rows:
                _reset_sequence(conn, "llm_predictions")

            conn.commit()

    if sync_exports and not is_processor_mode():
        from gex_core.exports import EXPORT_DIR
        from gex_core.storage import clear_sync_cache, sync_ticker_exports

        tickers = _tickers_from_exports(EXPORT_DIR)
        if not tickers:
            tickers_env = os.environ.get("GEX_DEFAULT_TICKERS") or os.environ.get("TICKERS") or "SPX"
            tickers = [t.strip().upper() for t in tickers_env.split(",") if t.strip()]
        clear_sync_cache()
        for ticker in tickers:
            stats.export_sync_added += sync_ticker_exports(ticker, EXPORT_DIR, force=True)

    return stats
