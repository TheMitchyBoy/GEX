"""Tests for Postgres bootstrap helpers."""

from unittest.mock import patch

from gex_core.bootstrap_data import backfill_min_snapshots, needs_postgres_bootstrap


def test_backfill_min_snapshots_default():
    assert backfill_min_snapshots() == 30


def test_needs_postgres_bootstrap_when_sparse(monkeypatch):
    monkeypatch.setenv("GEX_BACKFILL_MIN_SNAPSHOTS", "10")
    monkeypatch.setattr("gex_core.db.use_postgres", lambda: True)
    monkeypatch.setattr("gex_core.storage.count_snapshots", lambda ticker: 2)
    assert needs_postgres_bootstrap("SPX") is True


def test_needs_postgres_bootstrap_when_enough(monkeypatch):
    monkeypatch.setenv("GEX_BACKFILL_MIN_SNAPSHOTS", "10")
    monkeypatch.setattr("gex_core.db.use_postgres", lambda: True)
    monkeypatch.setattr("gex_core.storage.count_snapshots", lambda ticker: 50)
    assert needs_postgres_bootstrap("SPX") is False


def test_backfill_recent_intraday_empty_since_date_ignores_cursor(monkeypatch):
    from gex_core.intraday_backfill import backfill_recent_intraday

    monkeypatch.setattr("gex_core.processor_state.last_backfilled_date", lambda ticker: "2099-01-01")
    monkeypatch.setattr("gex_core.intraday_backfill.recent_market_dates", lambda days: ["2026-06-12", "2026-06-13"])
    monkeypatch.setattr("gex_core.intraday_backfill.backfill_intraday_minutes", lambda *a, **k: 1)
    monkeypatch.setattr("gex_core.processor_state.mark_backfilled_through", lambda *a, **k: None)

    results = backfill_recent_intraday("SPX", days=2, since_date="")
    assert results == {"2026-06-12": 1, "2026-06-13": 1}
