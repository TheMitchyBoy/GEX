"""Tests for snapshot anomaly detection."""

from gex_core.snapshot_anomalies import detect_snapshot_anomalies, maybe_dispatch_quality_alerts


def test_detect_snapshot_anomalies_low_quality():
    alerts = detect_snapshot_anomalies(
        ticker="SPX",
        ts="2026-06-15_143000",
        summary={"spot": 6000.0, "net_gamma_regime": "LONG gamma"},
        features={"quality_score": 0.4, "strike_count": 20},
        prior=None,
        validation={"ok": True, "status": "ok", "issues": [], "warnings": []},
    )
    assert any(a["title"] == "Low data quality score" for a in alerts)


def test_maybe_dispatch_quality_alerts_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GEX_QUALITY_ALERTS", raising=False)
    assert maybe_dispatch_quality_alerts("SPX", [{"severity": "high", "title": "x", "detail": "y"}]) is None
