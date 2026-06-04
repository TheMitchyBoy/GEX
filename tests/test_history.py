from pathlib import Path

from gex_core.history import collect_snapshot_files, get_latest_ts


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
