import gex_core.exports as exports
from scripts.train_gex_model import build_dataset


def test_build_dataset_uses_full_catalog_when_lookback_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(exports, "EXPORT_DIR", tmp_path)
    monkeypatch.setattr("scripts.train_gex_model.EXPORT_DIR", tmp_path)
    for idx, spot in enumerate((5000.0, 5001.0, 5002.0, 5003.0)):
        ts = f"2026-06-0{idx + 1}_120000"
        (tmp_path / f"SPX_gex_by_strike_{ts}.csv").write_text(
            f"strike,gex_bn_per_pct\n{5000 + idx},1.0\n",
            encoding="utf-8",
        )
        (tmp_path / f"SPX_summary_{ts}.json").write_text(
            f'{{"spot": {spot}, "market_date": "2026-06-0{idx + 1}"}}',
            encoding="utf-8",
        )

    df, stats = build_dataset("SPX", lookback_days=0, dedupe_identical_strikes=False)
    assert stats["catalog_timestamps"] == 4
    assert stats["window_timestamps"] == 4
    assert stats["training_rows"] == 4
    assert len(df) == 3


def test_build_dataset_can_dedupe_identical_strikes(tmp_path, monkeypatch):
    monkeypatch.setattr(exports, "EXPORT_DIR", tmp_path)
    monkeypatch.setattr("scripts.train_gex_model.EXPORT_DIR", tmp_path)
    for idx in range(3):
        ts = f"2026-06-05_{12 + idx}0000"
        (tmp_path / f"SPX_gex_by_strike_{ts}.csv").write_text(
            "strike,gex_bn_per_pct\n5000,1.0\n5010,2.0\n",
            encoding="utf-8",
        )
        (tmp_path / f"SPX_summary_{ts}.json").write_text(
            '{"spot": 5005, "market_date": "2026-06-05"}',
            encoding="utf-8",
        )

    df, stats = build_dataset("SPX", lookback_days=0, dedupe_identical_strikes=True)
    assert stats["training_rows"] == 1
    assert stats["skipped_identical_strikes"] == 2
    assert df.empty
