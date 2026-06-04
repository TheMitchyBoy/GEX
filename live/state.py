"""Persist live flow aggregator state to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_STATE_PATH = Path("data/live_flow_state.json")


def save_aggregator_state(aggregator, path: Path | None = None) -> None:
    path = path or DEFAULT_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "spot": aggregator.spot,
        "gex_by_strike": {str(k): v for k, v in aggregator.gex_by_strike.items()},
    }
    if hasattr(aggregator, "state"):
        payload["enhanced_state"] = {
            str(k): v for k, v in getattr(aggregator, "state", {}).items()
        }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_aggregator_state(path: Path | None = None) -> dict[str, Any] | None:
    path = path or DEFAULT_STATE_PATH
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def restore_gex_aggregator(aggregator, path: Path | None = None) -> None:
    data = load_aggregator_state(path)
    if not data:
        return
    if data.get("spot") is not None:
        aggregator.spot = float(data["spot"])
    for strike, gex in (data.get("gex_by_strike") or {}).items():
        aggregator.gex_by_strike[int(strike)] = float(gex)
    if hasattr(aggregator, "state") and data.get("enhanced_state"):
        aggregator.state = {int(k): v for k, v in data["enhanced_state"].items()}
