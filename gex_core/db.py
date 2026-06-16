"""Database abstraction: SQLite locally, PostgreSQL on Railway (DATABASE_URL)."""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_PG_INIT_LOCK = threading.Lock()
_PG_INITIALIZED = False

INDEX_SCHEMA_SQLITE = """
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

INDEX_SCHEMA_PG = """
        CREATE TABLE IF NOT EXISTS snapshots (
            ticker TEXT NOT NULL,
            ts TEXT NOT NULL,
            market_date TEXT,
            spot DOUBLE PRECISION,
            total_gex DOUBLE PRECISION,
            regime TEXT,
            summary_path TEXT,
            strike_path TEXT,
            indexed_at TEXT,
            summary_json JSONB,
            expiration_json JSONB,
            surface_json JSONB,
            greek_exposure_json JSONB,
            PRIMARY KEY (ticker, ts)
        );
        CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_ts
            ON snapshots (ticker, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_date
            ON snapshots (ticker, market_date);
        """

JOURNAL_SCHEMA_SQLITE = """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            option_type TEXT NOT NULL,
            strike REAL NOT NULL,
            qty REAL NOT NULL DEFAULT 1,
            entry_ts TEXT NOT NULL,
            exit_ts TEXT,
            entry_spot REAL NOT NULL,
            exit_spot REAL,
            entry_premium REAL NOT NULL,
            exit_premium REAL,
            pnl_pct REAL,
            pnl_usd REAL,
            exit_reason TEXT,
            signal_type TEXT,
            signal_strike REAL,
            signal_gamma REAL,
            gamma_delta REAL,
            ai_confidence REAL,
            ai_reason TEXT,
            meta_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status, entry_ts DESC);
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            payload_json TEXT,
            ai_verdict TEXT,
            ai_notes TEXT
        );
        CREATE TABLE IF NOT EXISTS trader_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """

JOURNAL_SCHEMA_PG = """
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            ticker TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            option_type TEXT NOT NULL,
            strike DOUBLE PRECISION NOT NULL,
            qty DOUBLE PRECISION NOT NULL DEFAULT 1,
            entry_ts TEXT NOT NULL,
            exit_ts TEXT,
            entry_spot DOUBLE PRECISION NOT NULL,
            exit_spot DOUBLE PRECISION,
            entry_premium DOUBLE PRECISION NOT NULL,
            exit_premium DOUBLE PRECISION,
            pnl_pct DOUBLE PRECISION,
            pnl_usd DOUBLE PRECISION,
            exit_reason TEXT,
            signal_type TEXT,
            signal_strike DOUBLE PRECISION,
            signal_gamma DOUBLE PRECISION,
            gamma_delta DOUBLE PRECISION,
            ai_confidence DOUBLE PRECISION,
            ai_reason TEXT,
            meta_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status, entry_ts DESC);
        CREATE TABLE IF NOT EXISTS decisions (
            id SERIAL PRIMARY KEY,
            ts TEXT NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            payload_json TEXT,
            ai_verdict TEXT,
            ai_notes TEXT
        );
        CREATE TABLE IF NOT EXISTS trader_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """

PREDICTION_SCHEMA_SQLITE = """
        CREATE TABLE IF NOT EXISTS llm_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            snapshot_ts TEXT,
            market_date TEXT,
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            payload_json TEXT NOT NULL,
            actual_json TEXT,
            outcome_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_llm_predictions_open
            ON llm_predictions (ticker, resolved_at, created_at DESC);
        """

PREDICTION_SCHEMA_PG = """
        CREATE TABLE IF NOT EXISTS llm_predictions (
            id SERIAL PRIMARY KEY,
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            snapshot_ts TEXT,
            market_date TEXT,
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            payload_json TEXT NOT NULL,
            actual_json TEXT,
            outcome_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_llm_predictions_open
            ON llm_predictions (ticker, resolved_at, created_at DESC);
        """

INSIGHT_SCHEMA_SQLITE = """
        CREATE TABLE IF NOT EXISTS daily_insights (
            ticker TEXT NOT NULL,
            market_date TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (ticker, market_date, kind)
        );
        CREATE INDEX IF NOT EXISTS idx_daily_insights_ticker_date
            ON daily_insights (ticker, market_date DESC);
        """

INSIGHT_SCHEMA_PG = """
        CREATE TABLE IF NOT EXISTS daily_insights (
            ticker TEXT NOT NULL,
            market_date TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (ticker, market_date, kind)
        );
        CREATE INDEX IF NOT EXISTS idx_daily_insights_ticker_date
            ON daily_insights (ticker, market_date DESC);
        """

STRIKES_SCHEMA_SQLITE = """
        CREATE TABLE IF NOT EXISTS snapshot_strikes (
            ticker TEXT NOT NULL,
            ts TEXT NOT NULL,
            strike REAL NOT NULL,
            gex_bn_per_pct REAL,
            cumulative_gex_bn_per_pct REAL,
            PRIMARY KEY (ticker, ts, strike)
        );
        CREATE INDEX IF NOT EXISTS idx_snapshot_strikes_ticker_ts
            ON snapshot_strikes (ticker, ts);
        """

STRIKES_SCHEMA_PG = """
        CREATE TABLE IF NOT EXISTS snapshot_strikes (
            ticker TEXT NOT NULL,
            ts TEXT NOT NULL,
            strike DOUBLE PRECISION NOT NULL,
            gex_bn_per_pct DOUBLE PRECISION,
            cumulative_gex_bn_per_pct DOUBLE PRECISION,
            PRIMARY KEY (ticker, ts, strike)
        );
        CREATE INDEX IF NOT EXISTS idx_snapshot_strikes_ticker_ts
            ON snapshot_strikes (ticker, ts);
        """

_SCHEMA_BY_GROUP: dict[str, tuple[str, str]] = {
    "index": (INDEX_SCHEMA_SQLITE, INDEX_SCHEMA_PG),
    "journal": (JOURNAL_SCHEMA_SQLITE, JOURNAL_SCHEMA_PG),
    "predictions": (PREDICTION_SCHEMA_SQLITE, PREDICTION_SCHEMA_PG),
    "insights": (INSIGHT_SCHEMA_SQLITE, INSIGHT_SCHEMA_PG),
    "strikes": (STRIKES_SCHEMA_SQLITE, STRIKES_SCHEMA_PG),
}

_POSTGRES_MIGRATIONS = (
    "ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS summary_json JSONB",
    "ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS expiration_json JSONB",
    "ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS surface_json JSONB",
    "ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS greek_exposure_json JSONB",
    "CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_date ON snapshots (ticker, market_date)",
)


class DatabaseError(Exception):
    """Raised for database failures across backends."""


class DbRow(dict):
    """Dict-like row compatible with sqlite3.Row access patterns."""

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


def _to_dbrow(row: Any, cursor: Any, *, postgres: bool) -> DbRow:
    if isinstance(row, dict):
        return DbRow(row)
    if hasattr(row, "keys"):
        return DbRow(dict(row))
    if postgres and cursor.description:
        cols = [desc[0] for desc in cursor.description]
        return DbRow(dict(zip(cols, row)))
    return DbRow(row)


class DbCursor:
    def __init__(self, cursor: Any, *, postgres: bool) -> None:
        self._cursor = cursor
        self._postgres = postgres
        self.lastrowid: int | None = None

    def fetchone(self) -> DbRow | None:
        row = self._cursor.fetchone()
        if row is None:
            return None
        return _to_dbrow(row, self._cursor, postgres=self._postgres)

    def fetchall(self) -> list[DbRow]:
        return [_to_dbrow(row, self._cursor, postgres=self._postgres) for row in self._cursor.fetchall()]


class DbConnection:
    """Thin wrapper over sqlite3 or psycopg connections with unified execute()."""

    def __init__(self, conn: Any, *, postgres: bool) -> None:
        self._conn = conn
        self._postgres = postgres

    @property
    def postgres(self) -> bool:
        return self._postgres

    def execute(self, sql: str, params: tuple | list = ()) -> DbCursor:
        sql = adapt_sql(sql, self._postgres)
        if self._postgres:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            wrapper = DbCursor(cur, postgres=True)
            if "RETURNING" in sql.upper():
                row = cur.fetchone()
                if row is not None:
                    wrapper.lastrowid = int(row["id"] if isinstance(row, dict) else row[0])
            return wrapper
        cur = self._conn.execute(sql, params)
        wrapper = DbCursor(cur, postgres=False)
        wrapper.lastrowid = cur.lastrowid
        return wrapper

    def executemany(self, sql: str, params_list: list[tuple | list]) -> None:
        sql = adapt_sql(sql, self._postgres)
        if self._postgres:
            with self._conn.cursor() as cur:
                cur.executemany(sql, params_list)
        else:
            self._conn.executemany(sql, params_list)

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DbConnection:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None and not self._postgres:
            self._conn.commit()
        self.close()


def database_url() -> str | None:
    """Return a psycopg-compatible Postgres URL when configured."""
    raw = os.environ.get("DATABASE_URL", "").strip()
    if not raw:
        return None
    if raw.startswith("postgres://"):
        return "postgresql://" + raw[len("postgres://") :]
    return raw


def use_postgres() -> bool:
    return database_url() is not None


def adapt_sql(sql: str, postgres: bool) -> str:
    if not postgres:
        return sql
    return sql.replace("?", "%s")


def insert_returning_id(conn: DbConnection, sql: str, params: tuple | list) -> int:
    """Insert a row and return its auto-generated id."""
    if conn.postgres:
        base = sql.rstrip().rstrip(";")
        if "RETURNING" not in base.upper():
            base = f"{base} RETURNING id"
        cur = conn.execute(base, params)
        row = cur.fetchone()
        if row is None:
            raise DatabaseError("INSERT did not return an id")
        return int(row["id"])
    cur = conn.execute(sql, params)
    if cur.lastrowid is None:
        raise DatabaseError("INSERT did not return an id")
    return int(cur.lastrowid)


def _sqlite_schema_sql(group: str) -> str:
    sqlite_sql, _ = _SCHEMA_BY_GROUP[group]
    return sqlite_sql


def _postgres_schema_statements() -> list[str]:
    parts = [
        _SCHEMA_BY_GROUP[group][1]
        for group in ("index", "journal", "predictions", "insights", "strikes")
    ]
    statements: list[str] = []
    for block in parts:
        for stmt in block.split(";"):
            cleaned = stmt.strip()
            if cleaned:
                statements.append(cleaned)
    statements.extend(_POSTGRES_MIGRATIONS)
    return statements


def postgres_schema_ddl() -> str:
    """Full PostgreSQL DDL for all application tables (idempotent)."""
    return ";\n\n".join(_postgres_schema_statements()) + ";\n"


def list_postgres_tables() -> list[str]:
    """Return public table names from the configured PostgreSQL database."""
    url = database_url()
    if not url:
        return []
    import psycopg

    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename
                """
            )
            return [str(row[0]) for row in cur.fetchall()]


POSTGRES_TABLES = (
    "snapshots",
    "snapshot_strikes",
    "trades",
    "decisions",
    "trader_state",
    "llm_predictions",
    "daily_insights",
)


def ensure_postgres_schema() -> list[str]:
    """Create all application tables in Postgres (idempotent). Returns table names."""
    global _PG_INITIALIZED
    url = database_url()
    if not url:
        return []
    with _PG_INIT_LOCK:
        if _PG_INITIALIZED:
            return list_postgres_tables()
        import psycopg

        logger.info("Initializing PostgreSQL schema")
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                for stmt in _postgres_schema_statements():
                    cur.execute(stmt)
            conn.commit()
        _PG_INITIALIZED = True
    return list_postgres_tables()


@contextmanager
def get_connection(
    *,
    group: str,
    sqlite_path: Path,
    sqlite_path_override: Path | None = None,
) -> Iterator[DbConnection]:
    """Open a DB connection for the given schema group.

    Uses PostgreSQL when DATABASE_URL is set, unless sqlite_path_override is
    provided (tests and explicit SQLite paths).
    """
    if sqlite_path_override is not None or not use_postgres():
        from gex_core.sqlite_util import connect_sqlite

        path = sqlite_path_override or sqlite_path
        conn = connect_sqlite(path, schema_sql=_sqlite_schema_sql(group))
        try:
            yield DbConnection(conn, postgres=False)
        finally:
            conn.close()
        return

    ensure_postgres_schema()
    import psycopg
    from psycopg.rows import dict_row

    conn = psycopg.connect(database_url(), row_factory=dict_row)
    try:
        yield DbConnection(conn, postgres=True)
    finally:
        conn.close()
