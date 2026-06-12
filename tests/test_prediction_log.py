import json

from gex_core.prediction_log import (
    calibrated_llm_confidence,
    get_llm_calibration_stats,
    log_llm_prediction,
    reconcile_llm_predictions,
)


def test_log_and_reconcile_prediction(tmp_path, monkeypatch):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    db = tmp_path / "journal.db"
    monkeypatch.setenv("GEX_TRADING_DB", str(db))
    monkeypatch.setattr("gex_core.exports.EXPORT_DIR", export_dir)
    monkeypatch.setattr("gex_core.history.EXPORT_DIR", export_dir)

    ts1 = "2026-06-04_100000"
    ts2 = "2026-06-04_110000"
    for ts, gex in ((ts1, 1.0), (ts2, 1.5)):
        (export_dir / f"SPX_gex_by_strike_{ts}.csv").write_text(
            f"strike,gex_bn_per_pct\n6000,{gex}\n",
            encoding="utf-8",
        )
        (export_dir / f"SPX_cumulative_gex_{ts}.csv").write_text(
            f"strike,cumulative_gex_bn_per_pct\n6000,{gex}\n",
            encoding="utf-8",
        )
        summary = {
            "spot": 6000,
            "total_gex_bn_per_pct": gex,
            "net_gamma_regime": "LONG gamma",
            "market_date": "2026-06-04",
        }
        (export_dir / f"SPX_summary_{ts}.json").write_text(json.dumps(summary), encoding="utf-8")

    log_llm_prediction(
        ticker="SPX",
        source="llm",
        snapshot_ts=ts1,
        prediction={
            "predicted_delta_gex_bn": 0.5,
            "spot_bias": "bullish",
            "confidence": 0.7,
            "llm_enhanced": True,
        },
    )
    resolved = reconcile_llm_predictions("SPX")
    assert resolved == 1
    stats = get_llm_calibration_stats("SPX", min_samples=1)
    assert stats["n"] == 1
    assert stats["sign_accuracy"] == 1.0
    assert calibrated_llm_confidence(0.8, "SPX") <= 1.0
