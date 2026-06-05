import json

from gex_core.exports import EXPORT_DIR
from gex_core.exports import list_export_timestamps
from gex_core.storage import (
    latest_timestamp,
    list_indexed_timestamps,
    prune_stale_index_entries,
    sync_ticker_exports,
    upsert_snapshot,
)


def test_upsert_and_latest_timestamp(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    monkeypatch.setenv("GEX_INDEX_DB", str(db))
    upsert_snapshot("SPX", "2026-06-01_120000", spot=5000.0, total_gex=1.5, regime="LONG gamma")
    assert latest_timestamp("SPX", path=db) == "2026-06-01_120000"


def test_sync_from_exports(tmp_path, monkeypatch):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    ts = "2026-06-02_000000"
    (export_dir / f"SPX_gex_by_strike_{ts}.csv").write_text("strike,gex\n4800,1.0\n", encoding="utf-8")
    (export_dir / f"SPX_cumulative_gex_{ts}.csv").write_text("strike,gex\n4800,1.0\n", encoding="utf-8")
    summary = {
        "spot": 5000,
        "total_gex_bn_per_pct": 2.0,
        "net_gamma_regime": "LONG gamma",
        "market_date": "2026-06-02",
    }
    (export_dir / f"SPX_summary_{ts}.json").write_text(json.dumps(summary), encoding="utf-8")

    db = tmp_path / "index.db"
    monkeypatch.setenv("GEX_INDEX_DB", str(db))
    added = sync_ticker_exports("SPX", export_dir)
    assert added == 1
    assert latest_timestamp("SPX", export_dir=export_dir, path=db) == ts


def test_prune_stale_index_entries(tmp_path, monkeypatch):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    db = tmp_path / "index.db"
    monkeypatch.setenv("GEX_INDEX_DB", str(db))

    upsert_snapshot("SPX", "2026-06-01_120000", strike_path=str(export_dir / "missing.csv"), path=db)
    upsert_snapshot("SPX", "2026-06-02_000000", strike_path=str(export_dir / "present.csv"), path=db)
    ts = "2026-06-02_000000"
    (export_dir / f"SPX_gex_by_strike_{ts}.csv").write_text("strike,gex\n4800,1.0\n", encoding="utf-8")

    pruned = prune_stale_index_entries("SPX", export_dir, path=db)
    assert pruned == 1
    assert list_indexed_timestamps("SPX", path=db) == [ts]
    assert list_export_timestamps("SPX", export_dir) == [ts]
