from pathlib import Path

import pandas as pd

from gex_core.history import build_index_timeline_history, load_snapshot_at_ts
from gex_core.storage import fetch_index_spot_series, upsert_snapshot


def test_load_snapshot_at_ts_reads_export(tmp_path, monkeypatch):
    ts = "2026-06-05_120000"
    (tmp_path / f"SPX_gex_by_strike_{ts}.csv").write_text(
        "strike,gex_bn_per_pct\n5000,1.0\n5010,-0.5\n",
        encoding="utf-8",
    )
    (tmp_path / f"SPX_cumulative_gex_{ts}.csv").write_text(
        "strike,cumulative_gex_bn_per_pct\n5000,1.0\n5010,0.5\n",
        encoding="utf-8",
    )
    (tmp_path / f"SPX_summary_{ts}.json").write_text(
        '{"spot": 5000, "market_date": "2026-06-05", "total_gex_bn_per_pct": 0.5}',
        encoding="utf-8",
    )

    row = load_snapshot_at_ts("SPX", ts, export_dir=tmp_path)
    assert row is not None
    assert row["ts"] == ts
    assert float(row["spot"]) == 5000.0
    assert isinstance(row["strike"], pd.Series)


def test_fetch_index_spot_series_subsamples(tmp_path, monkeypatch):
    db = tmp_path / "index.db"
    monkeypatch.setenv("GEX_INDEX_DB", str(db))
    for ts, spot in (
        ("2026-06-05_143000", 5000.0),
        ("2026-06-05_143500", 5001.0),
        ("2026-06-05_144000", 5002.0),
    ):
        upsert_snapshot("SPX", ts, spot=spot, total_gex=1.0, path=db)

    series = fetch_index_spot_series("SPX", days=90, interval_minutes=10, export_dir=tmp_path)
    assert len(series) >= 1
    assert all(row["spot"] > 0 for row in series)


def test_build_index_timeline_history_returns_rows():
    rows = build_index_timeline_history("SPX", days=90, interval_minutes=10, max_points=100)
    assert rows
    assert all(row.get("spot") for row in rows)
