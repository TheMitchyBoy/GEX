import os
import stat

from gex_core.data_root import configure_data_paths, resolve_data_root
from gex_core.exports import EXPORT_DIR, refresh_export_dir
from gex_core.trading.journal import db_path, get_performance_summary


def test_configure_data_paths_falls_back_when_repo_data_readonly(tmp_path, monkeypatch):
    readonly = tmp_path / "readonly-data"
    readonly.mkdir()
    os.chmod(readonly, stat.S_IRUSR | stat.S_IXUSR)

    writable = tmp_path / "writable-data"
    monkeypatch.setenv("GEX_DATA_DIR", str(readonly))
    monkeypatch.setenv("GEX_DATA_PATHS_CONFIGURED", "")
    monkeypatch.delenv("GEX_EXPORT_DIR", raising=False)
    monkeypatch.delenv("GEX_TRADING_DB", raising=False)
    monkeypatch.delenv("GEX_INDEX_DB", raising=False)

    # Reset module flag for test isolation.
    import gex_core.data_root as data_root

    data_root._CONFIGURED = False

    root = configure_data_paths()
    assert root != readonly
    assert root.exists()
    assert os.environ["GEX_EXPORT_DIR"] == str(root / "exports")
    refresh_export_dir()
    assert EXPORT_DIR == root / "exports"


def test_journal_connects_on_writable_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("GEX_TRADING_DB", str(tmp_path / "journal.db"))
    summary = get_performance_summary("SPX")
    assert summary["total_trades"] == 0
    assert db_path().exists() or db_path().parent.exists()
