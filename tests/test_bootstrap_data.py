"""Tests for Postgres bootstrap helpers."""

from unittest.mock import patch

from gex_core.bootstrap_data import (
    backfill_min_snapshots,
    needs_postgres_bootstrap,
    needs_postgres_catchup,
    postgres_latest_market_date,
)


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


def test_needs_postgres_catchup_when_latest_before_today(monkeypatch):
    monkeypatch.setattr("gex_core.db.use_postgres", lambda: True)
    monkeypatch.setattr("gex_core.bootstrap_data.postgres_latest_market_date", lambda ticker: "2026-04-30")
    monkeypatch.setattr("gex_core.market_time.market_today", lambda: "2026-06-15")
    assert needs_postgres_catchup("SPX") is True


def test_needs_postgres_catchup_when_current(monkeypatch):
    monkeypatch.setattr("gex_core.db.use_postgres", lambda: True)
    monkeypatch.setattr("gex_core.bootstrap_data.postgres_latest_market_date", lambda ticker: "2026-06-15")
    monkeypatch.setattr("gex_core.market_time.market_today", lambda: "2026-06-15")
    assert needs_postgres_catchup("SPX") is False


def test_postgres_latest_market_date_from_timestamp(monkeypatch):
    monkeypatch.setattr("gex_core.storage.latest_timestamp", lambda ticker: "2026-04-30_160000")
    assert postgres_latest_market_date("SPX") == "2026-04-30"


def test_sync_postgres_snapshots_runs_backfill_when_stale(monkeypatch):
    monkeypatch.setenv("GEX_STARTUP_BACKFILL", "1")
    monkeypatch.setattr("gex_core.db.use_postgres", lambda: True)
    monkeypatch.setattr("gex_core.db.ensure_postgres_schema", lambda: None)
    monkeypatch.setattr("gex_core.bootstrap_data._import_local_exports", lambda *a, **k: 0)
    monkeypatch.setattr("gex_core.storage.count_snapshots", lambda ticker: 120)
    monkeypatch.setattr("gex_core.bootstrap_data.needs_postgres_catchup", lambda ticker: True)
    monkeypatch.setattr("gex_core.bootstrap_data.postgres_latest_market_date", lambda ticker: "2026-04-30")
    monkeypatch.setattr(
        "gex_core.bootstrap_data._run_uw_backfill",
        lambda ticker, **kwargs: {"intraday_saved": 42, "daily_saved": 1, "since_date": "2026-04-30"},
    )

    from gex_core.bootstrap_data import sync_postgres_snapshots

    report = sync_postgres_snapshots("SPX")
    assert report["backfill_started"] is True
    assert report["intraday_saved"] == 42


def test_sync_postgres_snapshots_skips_when_current(monkeypatch):
    monkeypatch.setenv("GEX_STARTUP_BACKFILL", "1")
    monkeypatch.setattr("gex_core.db.use_postgres", lambda: True)
    monkeypatch.setattr("gex_core.db.ensure_postgres_schema", lambda: None)
    monkeypatch.setattr("gex_core.bootstrap_data._import_local_exports", lambda *a, **k: 0)
    monkeypatch.setattr("gex_core.storage.count_snapshots", lambda ticker: 120)
    monkeypatch.setattr("gex_core.bootstrap_data.needs_postgres_catchup", lambda ticker: False)
    monkeypatch.setattr("gex_core.bootstrap_data.postgres_latest_market_date", lambda ticker: "2026-06-15")

    from gex_core.bootstrap_data import sync_postgres_snapshots

    report = sync_postgres_snapshots("SPX")
    assert report["skipped"] is True
    assert report["reason"] == "up_to_date"


def test_backfill_recent_intraday_empty_since_date_ignores_cursor(monkeypatch):
    from gex_core.intraday_backfill import backfill_recent_intraday

    monkeypatch.setattr("gex_core.processor_state.last_backfilled_date", lambda ticker: "2099-01-01")
    monkeypatch.setattr("gex_core.intraday_backfill.recent_market_dates", lambda days: ["2026-06-12", "2026-06-13"])
    monkeypatch.setattr("gex_core.intraday_backfill.backfill_intraday_minutes", lambda *a, **k: 1)
    monkeypatch.setattr("gex_core.processor_state.mark_backfilled_through", lambda *a, **k: None)

    results = backfill_recent_intraday("SPX", days=2, since_date="")
    assert results == {"2026-06-12": 1, "2026-06-13": 1}
