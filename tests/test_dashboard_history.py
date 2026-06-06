from pathlib import Path

import pandas as pd

from gex_core.history import (
    build_gamma_levels_timeline,
    build_index_timeline_history,
    load_snapshot_at_ts,
    slim_gamma_timeline_rows,
)
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


def test_load_snapshot_metrics_includes_pos_gamma_peak_strike(tmp_path, monkeypatch):
    ts = "2026-06-05_120000"
    (tmp_path / f"SPX_gex_by_strike_{ts}.csv").write_text(
        "strike,gex_bn_per_pct\n5000,1.0\n5010,2.5\n5020,-0.5\n",
        encoding="utf-8",
    )
    (tmp_path / f"SPX_cumulative_gex_{ts}.csv").write_text(
        "strike,cumulative_gex_bn_per_pct\n5000,1.0\n5010,3.5\n5020,3.0\n",
        encoding="utf-8",
    )
    (tmp_path / f"SPX_summary_{ts}.json").write_text(
        '{"spot": 5005, "market_date": "2026-06-05", "total_gex_bn_per_pct": 3.0}',
        encoding="utf-8",
    )

    row = load_snapshot_at_ts("SPX", ts, export_dir=tmp_path)
    assert row is not None
    assert row["pos_gamma_peak_strike"] == 5010.0


def test_slim_gamma_timeline_rows_projects_chart_fields():
    history = [
        {
            "ts": "2026-06-05_120000",
            "ts_label": "2026-06-05 12:00:00",
            "spot": 5000.0,
            "pos_gamma_peak_strike": 5010.0,
            "gamma_flip": 4995.0,
            "call_wall": 5010.0,
            "put_wall": 4950.0,
            "total_gex": 1.0,
            "near_term_ratio": 0.2,
            "regime": "LONG gamma",
            "pos_gex": 2.0,
            "neg_gex": -1.0,
            "strike": "ignored",
        }
    ]
    rows = slim_gamma_timeline_rows(history)
    assert rows[0]["pos_gamma_peak_strike"] == 5010.0
    assert "strike" not in rows[0]


def test_build_gamma_levels_timeline_handles_missing_tail_spot():
    history = [
        {
            "ts": f"2026-06-05_{i:06d}",
            "ts_label": f"2026-06-05 {i:02d}:00:00",
            "spot": 5000.0 + i,
            "pos_gamma_peak_strike": 5010.0,
        }
        for i in range(120)
    ]
    history.append({"ts": "2026-06-05_999999", "ts_label": "tail", "spot": None})

    rows = build_gamma_levels_timeline("SPX", history=history, max_points=40)
    assert rows
    assert all(row.get("spot") for row in rows)


def test_build_gamma_levels_timeline_reuses_history():
    history = [
        {
            "ts": "2026-06-05_120000",
            "ts_label": "2026-06-05 12:00:00",
            "spot": 5005.0,
            "pos_gamma_peak_strike": 5010.0,
            "gamma_flip": 4995.0,
            "call_wall": 5010.0,
            "put_wall": 4950.0,
            "total_gex": 3.5,
            "near_term_ratio": 0.2,
            "regime": "LONG gamma",
            "pos_gex": 3.5,
            "neg_gex": 0.0,
        }
    ]

    rows = build_gamma_levels_timeline("SPX", history=history, max_points=10)
    assert len(rows) == 1
    assert rows[0]["spot"] == 5005.0
    assert rows[0]["pos_gamma_peak_strike"] == 5010.0
