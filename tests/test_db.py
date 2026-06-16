"""Tests for the database abstraction layer."""

from pathlib import Path

from gex_core.db import adapt_sql, get_connection, insert_returning_id, use_postgres


def test_adapt_sql_replaces_placeholders():
    assert adapt_sql("SELECT ? FROM t WHERE id = ?", postgres=True) == "SELECT %s FROM t WHERE id = %s"
    assert adapt_sql("SELECT ? FROM t", postgres=False) == "SELECT ? FROM t"


def test_use_postgres_false_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert use_postgres() is False


def test_use_postgres_true_with_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/gex")
    assert use_postgres() is True


def test_postgres_schema_ddl_includes_all_tables():
    from gex_core.db import POSTGRES_TABLES, postgres_schema_ddl

    ddl = postgres_schema_ddl()
    for table in POSTGRES_TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in ddl or table == "snapshots"
    assert "snapshot_strikes" in ddl
    assert "summary_json" in ddl


def test_sqlite_journal_insert_returning_id(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db = tmp_path / "journal.db"
    with get_connection(group="journal", sqlite_path=db) as conn:
        trade_id = insert_returning_id(
            conn,
            """
            INSERT INTO trades (
                ticker, status, option_type, strike, qty, entry_ts, entry_spot,
                entry_premium, signal_type, signal_strike, signal_gamma, gamma_delta,
                ai_confidence, ai_reason, meta_json
            ) VALUES (?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "SPX",
                "call",
                6000.0,
                1.0,
                "2026-06-01T12:00:00+00:00",
                5990.0,
                1.25,
                "max_positive_gamma",
                6000.0,
                1.5,
                0.5,
                0.7,
                "test",
                "{}",
            ),
        )
        conn.commit()
    assert trade_id == 1

    with get_connection(group="journal", sqlite_path=Path(db)) as conn:
        row = conn.execute("SELECT ticker FROM trades WHERE id = ?", (trade_id,)).fetchone()
    assert row is not None
    assert row["ticker"] == "SPX"
