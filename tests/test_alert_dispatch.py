from gex_core.alert_dispatch import filter_alerts_for_dispatch, maybe_dispatch_alerts


def test_filter_alerts_for_dispatch():
    alerts = [
        {"severity": "high", "title": "A"},
        {"severity": "low", "title": "B"},
    ]
    out = filter_alerts_for_dispatch(alerts, "high")
    assert len(out) == 1
    assert out[0]["title"] == "A"


def test_maybe_dispatch_skips_without_webhook(monkeypatch):
    monkeypatch.delenv("GEX_ALERT_WEBHOOK_URL", raising=False)
    status = maybe_dispatch_alerts("SPX", [{"severity": "high", "title": "x"}], manual=True)
    assert status is not None
    assert status["ok"] is False
