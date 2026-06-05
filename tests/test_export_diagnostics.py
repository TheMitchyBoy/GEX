import json

from gex_core.export_diagnostics import forecast_blocker_from_state, summarize_export_state


def test_forecast_blocker_when_csvs_missing_on_disk():
    msg = forecast_blocker_from_state(
        {
            "strike_csv_on_disk": 1,
            "forecast_loadable": 1,
            "indexed_before_sync": 1655,
            "collected_in_window": 1,
            "lookback_days": 30,
        },
        window_count=1,
    )
    assert "Strike CSV files on disk: 1" in msg
    assert "1654 stale" in msg or "1655 stale" in msg or "1655 timestamps" in msg
    assert "GEX_STARTUP_BACKFILL=1" in msg


def test_summarize_export_state_counts_files(tmp_path, monkeypatch):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    db = tmp_path / "index.db"
    monkeypatch.setenv("GEX_INDEX_DB", str(db))

    for i in range(1, 6):
        ts = f"2026-06-0{i}_12000{i}"
        (export_dir / f"SPX_gex_by_strike_{ts}.csv").write_text(
            "strike,gex_bn_per_pct\n5000,1.0\n",
            encoding="utf-8",
        )
        (export_dir / f"SPX_cumulative_gex_{ts}.csv").write_text(
            "strike,cumulative_gex_bn_per_pct\n5000,1.0\n",
            encoding="utf-8",
        )
        summary = {"spot": 5000, "total_gex_bn_per_pct": 1.0, "net_gamma_regime": "LONG gamma"}
        (export_dir / f"SPX_summary_{ts}.json").write_text(json.dumps(summary), encoding="utf-8")

    monkeypatch.setattr("gex_core.exports.EXPORT_DIR", export_dir)
    monkeypatch.setattr("gex_core.history.EXPORT_DIR", export_dir)
    monkeypatch.setattr("gex_core.storage.EXPORT_DIR", export_dir)

    state = summarize_export_state("SPX", lookback_days=0, max_snapshots=0)
    assert state["strike_csv_on_disk"] == 5
    assert state["forecast_loadable"] == 5
    assert state["needs_backfill"] is False
