from pathlib import Path
from datetime import date

from gex_core.history import build_history, collect_snapshot_files, get_latest_ts
from gex_core.refresh import recent_market_dates


def test_collect_snapshot_files_finds_exports():
    export_dir = Path("data/exports")
    if not export_dir.exists():
        return
    tickers = {p.name.split("_")[0] for p in export_dir.glob("SPX_*.csv")}
    if "SPX" not in tickers:
        return
    files = collect_snapshot_files("SPX", export_dir)
    assert files
    assert get_latest_ts("SPX", export_dir) in files


def test_build_history_skips_consecutive_duplicate_strike_profiles(tmp_path: Path):
    for ts in ["2026-06-03_000000", "2026-06-04_000000"]:
        (tmp_path / f"SPX_gex_by_strike_{ts}.csv").write_text(
            "strike,gex_bn_per_pct\n5000,1.0\n5010,2.0\n",
            encoding="utf-8",
        )
        (tmp_path / f"SPX_cumulative_gex_{ts}.csv").write_text(
            "strike,cumulative_gex_bn_per_pct\n5000,1.0\n5010,3.0\n",
            encoding="utf-8",
        )

    history = build_history("SPX", tmp_path)

    assert [row["ts"] for row in history] == ["2026-06-03_000000"]


def test_recent_market_dates_skips_weekends():
    dates = recent_market_dates(days=7, today=date(2026, 6, 5))

    assert dates == [
        "2026-06-01",
        "2026-06-02",
        "2026-06-03",
        "2026-06-04",
        "2026-06-05",
    ]
