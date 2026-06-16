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

ATM_STRIKES_SCHEMA_PG = """
        CREATE TABLE IF NOT EXISTS snapshot_strikes_atm (
            ticker TEXT NOT NULL,
            ts TEXT NOT NULL,
            strike DOUBLE PRECISION NOT NULL,
            gex_bn_per_pct DOUBLE PRECISION,
            cumulative_gex_bn_per_pct DOUBLE PRECISION,
            PRIMARY KEY (ticker, ts, strike)
        );
        CREATE INDEX IF NOT EXISTS idx_snapshot_strikes_atm_ticker_ts
            ON snapshot_strikes_atm (ticker, ts);
        """

FEATURES_SCHEMA_PG = """
        CREATE TABLE IF NOT EXISTS snapshot_features (
            ticker TEXT NOT NULL,
            ts TEXT NOT NULL,
            prior_ts TEXT,
            snapshot_at TIMESTAMPTZ,
            gamma_flip DOUBLE PRECISION,
            call_wall DOUBLE PRECISION,
            put_wall DOUBLE PRECISION,
            pos_gamma_peak_strike DOUBLE PRECISION,
            flip_distance_pct DOUBLE PRECISION,
            wall_spread DOUBLE PRECISION,
            gex_concentration DOUBLE PRECISION,
            near_term_ratio DOUBLE PRECISION,
            zero_dte_ratio DOUBLE PRECISION,
            term_curvature DOUBLE PRECISION,
            expiration_count DOUBLE PRECISION,
            front_term_ratio DOUBLE PRECISION,
            back_term_ratio DOUBLE PRECISION,
            delta_gex DOUBLE PRECISION,
            delta_spot DOUBLE PRECISION,
            spot_return DOUBLE PRECISION,
            regime_changed BOOLEAN,
            surface_vector JSONB,
            strike_profile_hash TEXT,
            strike_count INTEGER,
            PRIMARY KEY (ticker, ts)
        );
        CREATE INDEX IF NOT EXISTS idx_snapshot_features_ticker_ts
            ON snapshot_features (ticker, ts DESC);
        """

DIAGNOSTICS_SCHEMA_PG = """
        CREATE TABLE IF NOT EXISTS snapshot_diagnostics (
            ticker TEXT NOT NULL,
            ts TEXT NOT NULL,
            status TEXT NOT NULL,
            validation_json JSONB,
            uw_fetch_ms DOUBLE PRECISION,
            postgres_write_ms DOUBLE PRECISION,
            indexed_at TEXT,
            PRIMARY KEY (ticker, ts)
        );
        CREATE INDEX IF NOT EXISTS idx_snapshot_diagnostics_ticker_ts
            ON snapshot_diagnostics (ticker, ts DESC);
        """

PROCESSOR_STATE_SCHEMA_PG = """
        CREATE TABLE IF NOT EXISTS processor_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """

DAILY_QUALITY_SCHEMA_PG = """
        CREATE TABLE IF NOT EXISTS daily_quality_stats (
            ticker TEXT NOT NULL,
            market_date TEXT NOT NULL,
            payload_json JSONB NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (ticker, market_date)
        );
        CREATE INDEX IF NOT EXISTS idx_daily_quality_stats_ticker_date
            ON daily_quality_stats (ticker, market_date DESC);
        """

PREDICTION_ACCURACY_SCHEMA_PG = """
        CREATE TABLE IF NOT EXISTS prediction_accuracy_daily (
            ticker TEXT NOT NULL,
            market_date TEXT NOT NULL,
            payload_json JSONB NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (ticker, market_date)
        );
        CREATE INDEX IF NOT EXISTS idx_prediction_accuracy_daily_ticker_date
            ON prediction_accuracy_daily (ticker, market_date DESC);
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
    "ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS snapshot_at TIMESTAMPTZ",
    "ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS prior_ts TEXT",
    "CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_date ON snapshots (ticker, market_date)",
    "CREATE INDEX IF NOT EXISTS idx_snapshots_snapshot_at ON snapshots (ticker, snapshot_at DESC)",
    "ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION",
    "ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS flip_confidence TEXT",
    "ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS regime_consistent BOOLEAN",
    "ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS spot_source TEXT",
    "ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS spot_disagreement_pct DOUBLE PRECISION",
    "ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS strike_profile_confidence TEXT",
    "ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS data_lag_sec DOUBLE PRECISION",
    "ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS uw_rate_limit_json JSONB",
    "ALTER TABLE snapshot_diagnostics ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION",
    "ALTER TABLE snapshot_diagnostics ADD COLUMN IF NOT EXISTS data_lag_sec DOUBLE PRECISION",
    "ALTER TABLE snapshot_diagnostics ADD COLUMN IF NOT EXISTS uw_rate_limit_json JSONB",
)

_PROCESSOR_EXTRA_SCHEMA_BLOCKS = (
    ATM_STRIKES_SCHEMA_PG,
    FEATURES_SCHEMA_PG,
    DIAGNOSTICS_SCHEMA_PG,
    PROCESSOR_STATE_SCHEMA_PG,
    DAILY_QUALITY_SCHEMA_PG,
    PREDICTION_ACCURACY_SCHEMA_PG,
)

_LATEST_SNAPSHOT_VIEW_SQL = """
CREATE MATERIALIZED VIEW IF NOT EXISTS latest_snapshot AS
SELECT DISTINCT ON (ticker)
    ticker, ts, market_date, spot, total_gex, regime, indexed_at, snapshot_at, prior_ts
FROM snapshots
ORDER BY ticker, ts DESC
"""

_LATEST_SNAPSHOT_INDEX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_latest_snapshot_ticker ON latest_snapshot (ticker)"
)

_TRAINING_SNAPSHOTS_VIEW_SQL = """
CREATE OR REPLACE VIEW training_snapshots AS
SELECT
    s.ticker,
    s.ts,
    s.market_date,
    s.spot,
    s.total_gex,
    s.regime,
    s.snapshot_at,
    f.quality_score,
    f.flip_confidence,
    f.regime_consistent,
    f.strike_count,
    f.delta_gex,
    f.spot_return,
    d.status AS diagnostic_status
FROM snapshots s
JOIN snapshot_features f ON f.ticker = s.ticker AND f.ts = s.ts
LEFT JOIN snapshot_diagnostics d ON d.ticker = s.ticker AND d.ts = s.ts
WHERE COALESCE(d.status, 'ok') IN ('ok', 'ok_with_warnings')
  AND COALESCE(f.quality_score, 0) >= 0.8
  AND COALESCE(f.strike_profile_confidence, 'high') <> 'low'
  AND COALESCE(s.summary_json->>'strike_profile_source', '') <> 'eod_scaled'
"""


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


def _postgres_schema_statements(*, processor_only: bool = False) -> list[str]:
    groups = ("index", "strikes") if processor_only else ("index", "journal", "predictions", "insights", "strikes")
    parts = [_SCHEMA_BY_GROUP[group][1] for group in groups]
    if processor_only:
        parts.extend(_PROCESSOR_EXTRA_SCHEMA_BLOCKS)
    statements: list[str] = []
    for block in parts:
        for stmt in block.split(";"):
            cleaned = stmt.strip()
            if cleaned:
                statements.append(cleaned)
    statements.extend(_POSTGRES_MIGRATIONS)
    return statements


def refresh_latest_snapshot_view() -> None:
    """Refresh the latest_snapshot materialized view (no-op without Postgres)."""
    url = database_url()
    if not url:
        return
    import psycopg

    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(_LATEST_SNAPSHOT_VIEW_SQL)
            cur.execute(_LATEST_SNAPSHOT_INDEX_SQL)
            try:
                cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY latest_snapshot")
            except Exception:
                cur.execute("REFRESH MATERIALIZED VIEW latest_snapshot")
        conn.commit()


def postgres_schema_ddl(*, processor_only: bool = False) -> str:
    """Full PostgreSQL DDL for application tables (idempotent)."""
    return ";\n\n".join(_postgres_schema_statements(processor_only=processor_only)) + ";\n"


PROCESSOR_POSTGRES_TABLES = (
    "snapshots",
    "snapshot_strikes",
    "snapshot_strikes_atm",
    "snapshot_features",
    "snapshot_diagnostics",
    "processor_state",
    "daily_quality_stats",
    "prediction_accuracy_daily",
)


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


def ensure_postgres_schema(*, processor_only: bool | None = None) -> list[str]:
    """Create application tables in Postgres (idempotent). Returns table names."""
    global _PG_INITIALIZED
    if processor_only is None:
        from gex_core.runtime_mode import is_processor_mode

        processor_only = is_processor_mode()
    url = database_url()
    if not url:
        return []
    with _PG_INIT_LOCK:
        if _PG_INITIALIZED:
            return list_postgres_tables()
        import psycopg

        logger.info("Initializing PostgreSQL schema (processor_only=%s)", processor_only)
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                for stmt in _postgres_schema_statements(processor_only=processor_only):
                    cur.execute(stmt)
                if processor_only:
                    cur.execute(_LATEST_SNAPSHOT_VIEW_SQL)
                    cur.execute(_LATEST_SNAPSHOT_INDEX_SQL)
                    cur.execute(_TRAINING_SNAPSHOTS_VIEW_SQL)
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
