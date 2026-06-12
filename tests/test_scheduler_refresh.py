import threading
import time

import web_app


def test_run_scheduler_job_skips_when_previous_still_running():
    web_app._refresh_job_lock.acquire()
    try:
        calls = {"n": 0}

        def _work():
            calls["n"] += 1

        web_app._run_scheduler_job("test-refresh", web_app._refresh_job_lock, _work)
        time.sleep(0.05)
        assert calls["n"] == 0
    finally:
        web_app._refresh_job_lock.release()


def test_run_scheduler_job_runs_in_background_thread():
    done = threading.Event()

    def _work():
        done.set()

    web_app._run_scheduler_job("test-refresh", web_app._refresh_job_lock, _work)
    assert done.wait(timeout=2.0)


def test_scheduled_refresh_uses_light_alert_history(monkeypatch):
    captured: dict = {}

    def _fake_history(ticker):
        captured["ticker"] = ticker
        return [{"ts": "2026-06-12_120000", "regime": "LONG gamma"}]

    monkeypatch.setattr(web_app, "_scheduler_alert_history", _fake_history)
    monkeypatch.setattr(web_app, "maybe_dispatch_alerts", lambda *a, **k: captured.setdefault("dispatched", True))
    monkeypatch.setattr(web_app, "generate_alerts", lambda *a, **k: [])
    monkeypatch.setattr(web_app, "predict_next_snapshot", lambda *a, **k: {})
    monkeypatch.setattr(web_app, "_select_snapshot", lambda history, _ts: history[-1])

    web_app._auto_dispatch_alerts("SPX")
    assert captured["ticker"] == "SPX"
    assert captured.get("dispatched") is True
