"""Security tests for webhook SSRF validation and manual dispatch auth."""

import pytest

from gex_core.intelligence import validate_webhook_url


def test_rejects_non_https():
    ok, reason = validate_webhook_url("http://example.com/hook")
    assert ok is False
    assert "https" in reason.lower()


def test_rejects_loopback(monkeypatch):
    monkeypatch.delenv("GEX_ALERT_ALLOW_INSECURE", raising=False)
    ok, _ = validate_webhook_url("https://127.0.0.1/hook")
    assert ok is False


def test_rejects_link_local_metadata(monkeypatch):
    monkeypatch.delenv("GEX_ALERT_ALLOW_INSECURE", raising=False)
    ok, reason = validate_webhook_url("https://169.254.169.254/latest/meta-data")
    assert ok is False
    assert "non-public" in reason.lower()


def test_rejects_private_range(monkeypatch):
    monkeypatch.delenv("GEX_ALERT_ALLOW_INSECURE", raising=False)
    ok, _ = validate_webhook_url("https://10.0.0.5/hook")
    assert ok is False


def test_allows_public_https(monkeypatch):
    monkeypatch.delenv("GEX_ALERT_ALLOW_INSECURE", raising=False)
    # 8.8.8.8 is a public address; getaddrinfo on a literal IP does not hit DNS.
    ok, reason = validate_webhook_url("https://8.8.8.8/hook")
    assert ok is True, reason


def test_allow_insecure_opt_out(monkeypatch):
    monkeypatch.setenv("GEX_ALERT_ALLOW_INSECURE", "1")
    ok, _ = validate_webhook_url("http://127.0.0.1/hook")
    assert ok is True


def test_dispatch_rejects_unsafe_url(monkeypatch):
    from gex_core.intelligence import dispatch_alerts_to_webhook

    monkeypatch.delenv("GEX_ALERT_ALLOW_INSECURE", raising=False)
    monkeypatch.setenv("GEX_ALERT_WEBHOOK_URL", "http://169.254.169.254/")
    ok, reason = dispatch_alerts_to_webhook("SPX", [{"severity": "high", "title": "x"}])
    assert ok is False
    assert "rejected" in reason.lower()


def test_manual_dispatch_requires_token(monkeypatch):
    monkeypatch.setenv("GEX_DISABLE_SCHEDULER", "1")
    import web_app

    class _Req:
        def __init__(self, args=None, headers=None):
            self.args = args or {}
            self.headers = headers or {}

    monkeypatch.delenv("GEX_ADMIN_TOKEN", raising=False)
    assert web_app._admin_action_authorized(_Req(args={"admin_token": "x"})) is False

    monkeypatch.setenv("GEX_ADMIN_TOKEN", "s3cret")
    assert web_app._admin_action_authorized(_Req()) is False
    assert web_app._admin_action_authorized(_Req(args={"admin_token": "wrong"})) is False
    assert web_app._admin_action_authorized(_Req(args={"admin_token": "s3cret"})) is True
    assert web_app._admin_action_authorized(_Req(headers={"X-Admin-Token": "s3cret"})) is True


def test_dispatch_alerts_route_requires_token(monkeypatch):
    monkeypatch.setenv("GEX_DISABLE_SCHEDULER", "1")
    import web_app

    client = web_app.APP.test_client()
    monkeypatch.setenv("GEX_ADMIN_TOKEN", "s3cret")
    denied = client.post("/ticker/SPX/dispatch-alerts")
    assert denied.status_code == 403

    monkeypatch.setattr(
        web_app,
        "maybe_dispatch_alerts",
        lambda *_a, **_k: {"ok": True, "message": "sent", "dispatched": True},
    )
    ok = client.post(
        "/ticker/SPX/dispatch-alerts",
        headers={"X-Admin-Token": "s3cret"},
    )
    assert ok.status_code == 200
    assert ok.get_json()["dispatched"] is True
