import json

import gex_core.exports as exports
from gex_core.startup import should_retrain_on_start


def _write_exports(tmp_path, count: int = 4) -> None:
    for idx in range(count):
        ts = f"2026-06-05_{12 + idx}0000"
        (tmp_path / f"SPX_gex_by_strike_{ts}.csv").write_text(
            "strike,gex_bn_per_pct\n5000,1.0\n",
            encoding="utf-8",
        )


def test_should_retrain_when_manifest_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("GEX_RETRAIN_ON_START", "1")
    monkeypatch.setattr(exports, "EXPORT_DIR", tmp_path)
    monkeypatch.setattr("gex_core.models_manifest.MODELS_DIR", tmp_path / "models")
    _write_exports(tmp_path, 4)
    assert should_retrain_on_start("SPX") is True


def test_should_skip_retrain_when_manifest_covers_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("GEX_RETRAIN_ON_START", "1")
    monkeypatch.setattr(exports, "EXPORT_DIR", tmp_path)
    _write_exports(tmp_path, 4)
    models = tmp_path / "models" / "SPX"
    models.mkdir(parents=True)
    (models / "manifest.json").write_text(
        json.dumps({"catalog_timestamps": 4, "lookback_days": 0, "n_snapshots": 4}),
        encoding="utf-8",
    )
    monkeypatch.setattr("gex_core.models_manifest.MODELS_DIR", tmp_path / "models")
    assert should_retrain_on_start("SPX") is False
