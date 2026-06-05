"""Webhook alert dispatch with dedupe and cooldown."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from gex_core.alert_config import load_alert_config
from gex_core.intelligence import dispatch_alerts_to_webhook

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = _REPO_ROOT / "data" / ".alert_dispatch_state.json"

_SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        with STATE_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _alert_fingerprint(alerts: list[dict]) -> str:
    parts = [f"{a.get('severity')}:{a.get('title')}" for a in alerts]
    return "|".join(sorted(parts))


def filter_alerts_for_dispatch(alerts: list[dict], min_severity: str) -> list[dict]:
    floor = _SEVERITY_RANK.get(min_severity.lower(), 3)
    return [a for a in alerts if _SEVERITY_RANK.get(a.get("severity", "low"), 0) >= floor]


def maybe_dispatch_alerts(
    ticker: str,
    alerts: list[dict],
    *,
    manual: bool = False,
) -> dict | None:
    """Dispatch alerts when manual or auto rules pass. Returns status dict or None."""
    cfg = load_alert_config()
    if not manual and not cfg.auto_dispatch:
        return None

    to_send = alerts if manual else filter_alerts_for_dispatch(alerts, cfg.min_severity_for_auto)
    if not to_send:
        return {"ok": False, "message": "No alerts meet dispatch criteria.", "dispatched": False}

    if not manual:
        state = _load_state()
        entry = state.get(ticker.upper(), {})
        fp = _alert_fingerprint(to_send)
        if entry.get("fingerprint") == fp:
            return {
                "ok": False,
                "message": "Duplicate alert set suppressed (unchanged since last dispatch).",
                "dispatched": False,
            }
        last_at = entry.get("dispatched_at_utc")
        if last_at:
            try:
                prev = datetime.fromisoformat(last_at.replace("Z", "+00:00"))
                age_min = (datetime.now(timezone.utc) - prev).total_seconds() / 60.0
                if age_min < cfg.dispatch_cooldown_minutes:
                    return {
                        "ok": False,
                        "message": f"Cooldown active ({age_min:.0f}m < {cfg.dispatch_cooldown_minutes}m).",
                        "dispatched": False,
                    }
            except ValueError:
                pass

    ok, message = dispatch_alerts_to_webhook(ticker, to_send)
    if ok and not manual:
        state = _load_state()
        state[ticker.upper()] = {
            "fingerprint": _alert_fingerprint(to_send),
            "dispatched_at_utc": datetime.now(timezone.utc).isoformat(),
            "alert_count": len(to_send),
        }
        _save_state(state)
    return {"ok": ok, "message": message, "dispatched": ok}
