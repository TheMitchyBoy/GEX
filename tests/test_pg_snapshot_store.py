"""Tests for PostgreSQL snapshot payload storage."""

import json

import pandas as pd
import pytest

from gex_core.pg_snapshot_store import export_csv_enabled, write_snapshot_to_postgres


def test_export_csv_enabled_without_postgres(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert export_csv_enabled() is True


def test_export_csv_disabled_with_postgres_by_default(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/gex")
    monkeypatch.delenv("GEX_EXPORT_CSV", raising=False)
    assert export_csv_enabled() is False


def test_export_csv_opt_in_with_postgres(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/gex")
    monkeypatch.setenv("GEX_EXPORT_CSV", "1")
    assert export_csv_enabled() is True


def test_write_snapshot_to_postgres_noop_without_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    write_snapshot_to_postgres(
        "SPX",
        "2026-06-01_120000",
        gex_by_strike=pd.Series({6000.0: 1.5}),
        cumulative_gex=pd.Series({6000.0: 1.5}),
        summary={"spot": 6000.0, "total_gex_bn_per_pct": 1.5, "net_gamma_regime": "LONG gamma"},
    )


@pytest.mark.skipif(
    not __import__("os").environ.get("DATABASE_URL"),
    reason="DATABASE_URL required for integration test",
)
def test_write_snapshot_roundtrip_postgres():
    gex = pd.Series({6000.0: 1.5, 6050.0: 2.0})
    cumulative = pd.Series({6000.0: 1.5, 6050.0: 3.5})
    summary = {
        "market_date": "2026-06-01",
        "spot": 6000.0,
        "total_gex_bn_per_pct": 1.5,
        "net_gamma_regime": "LONG gamma",
        "gamma_flip": 5980.0,
    }
    write_snapshot_to_postgres(
        "SPX",
        "2026-06-01_120000",
        gex_by_strike=gex,
        cumulative_gex=cumulative,
        summary=summary,
    )

    from gex_core.db import database_url

    import psycopg

    with psycopg.connect(database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT spot, summary_json FROM snapshots WHERE ticker = %s AND ts = %s",
                ("SPX", "2026-06-01_120000"),
            )
            row = cur.fetchone()
            cur.execute(
                "SELECT COUNT(*) FROM snapshot_strikes WHERE ticker = %s AND ts = %s",
                ("SPX", "2026-06-01_120000"),
            )
            strike_count = cur.fetchone()[0]
    assert row is not None
    assert float(row[0]) == 6000.0
    payload = row[1] if isinstance(row[1], dict) else json.loads(row[1])
    assert payload["gamma_flip"] == 5980.0
    assert strike_count == 2
