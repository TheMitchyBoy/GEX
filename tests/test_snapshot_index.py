"""Tests for unified snapshot timestamp listing."""

from unittest.mock import patch

from gex_core.storage import count_snapshots, list_snapshot_timestamps


def test_list_snapshot_timestamps_prefers_postgres(monkeypatch):
    monkeypatch.setattr(
        "gex_core.storage.list_postgres_snapshot_timestamps",
        lambda ticker: ["2026-06-01_100000", "2026-06-01_110000"],
    )
    monkeypatch.setattr("gex_core.storage.list_indexed_timestamps", lambda *args, **kwargs: [])
    assert list_snapshot_timestamps("SPX") == ["2026-06-01_100000", "2026-06-01_110000"]
    assert count_snapshots("SPX") == 2


def test_list_snapshot_timestamps_falls_back_to_disk(monkeypatch, tmp_path):
    monkeypatch.setattr("gex_core.storage.list_postgres_snapshot_timestamps", lambda ticker: [])
    monkeypatch.setattr("gex_core.storage.list_indexed_timestamps", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "gex_core.storage.scan_export_timestamps",
        lambda ticker, export_dir: ["2026-06-02_120000"],
    )
    assert list_snapshot_timestamps("SPX", export_dir=tmp_path) == ["2026-06-02_120000"]
