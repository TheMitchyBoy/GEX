"""Tests for SQLite -> PostgreSQL migration helpers."""

import json
import sqlite3
from pathlib import Path

from gex_core.migrate_postgres import (
    MigrationStats,
    _read_sqlite_rows,
    _sqlite_row_count,
    migrate_sqlite_to_postgres,
    sqlite_has_data,
)


def _seed_index_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE snapshots (
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
            """
        )
        conn.execute(
            """
            INSERT INTO snapshots (
                ticker, ts, market_date, spot, total_gex, regime, summary_path, strike_path, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "SPX",
                "2026-06-01_120000",
                "2026-06-01",
                6000.0,
                1.5,
                "LONG gamma",
                "/tmp/summary.json",
                "/tmp/strike.csv",
                "2026-06-01T12:00:00+00:00",
            ),
        )
        conn.commit()


def _seed_journal_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE trader_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE llm_predictions (
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
            """
        )
        conn.execute(
            "INSERT INTO trader_state (key, value, updated_at) VALUES ('armed', '0', '2026-06-01T12:00:00+00:00')"
        )
        conn.execute(
            """
            INSERT INTO llm_predictions (
                ticker, source, snapshot_ts, market_date, created_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("SPX", "daily_strategy", "2026-06-01_120000", "2026-06-01", "2026-06-01T12:00:00+00:00", "{}"),
        )
        conn.commit()


def test_sqlite_has_data(tmp_path, monkeypatch):
    index_db = tmp_path / "gex_index.db"
    journal_db = tmp_path / "trading_journal.db"
    _seed_index_db(index_db)
    _seed_journal_db(journal_db)
    monkeypatch.setenv("GEX_INDEX_DB", str(index_db))
    monkeypatch.setenv("GEX_TRADING_DB", str(journal_db))
    assert sqlite_has_data() is True
    assert _sqlite_row_count(index_db, "snapshots") == 1
    assert len(_read_sqlite_rows(journal_db, "llm_predictions")) == 1


def test_migrate_requires_database_url(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    stats = migrate_sqlite_to_postgres()
    assert stats.skipped is True
    assert stats.reason == "DATABASE_URL is not set"


def test_migration_stats_total_rows():
    stats = MigrationStats(snapshots=2, trades=1, trader_state=1)
    assert stats.total_rows() == 4
