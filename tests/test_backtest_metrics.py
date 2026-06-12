from gex_core.backtest_metrics import backtest_delta_sign_accuracy, backtest_metrics_enabled
from gex_core.history import clear_history_cache


def test_backtest_metrics_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GEX_BACKTEST_METRICS", raising=False)
    assert backtest_metrics_enabled() is False
    result = backtest_delta_sign_accuracy("SPX")
    assert result["n"] == 0
    assert result["accuracy"] is None


def test_backtest_caps_walk_forward_folds(monkeypatch, tmp_path):
    monkeypatch.setenv("GEX_BACKTEST_METRICS", "1")
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    monkeypatch.setattr("gex_core.exports.EXPORT_DIR", export_dir)
    monkeypatch.setattr("gex_core.history.EXPORT_DIR", export_dir)
    monkeypatch.setenv("GEX_BACKTEST_MAX_FOLDS", "3")
    monkeypatch.setenv("GEX_BACKTEST_MAX_SNAPSHOTS", "12")

    for i in range(1, 13):
        ts = f"2026-06-{i:02d}_120000"
        (export_dir / f"SPX_gex_by_strike_{ts}.csv").write_text(
            f"strike,gex_bn_per_pct\n{5000 + i},{float(i)}\n",
            encoding="utf-8",
        )
        (export_dir / f"SPX_cumulative_gex_{ts}.csv").write_text(
            f"strike,cumulative_gex_bn_per_pct\n{5000 + i},{float(i)}\n",
            encoding="utf-8",
        )

    clear_history_cache()
    result = backtest_delta_sign_accuracy("SPX")
    assert result["n"] <= 3
