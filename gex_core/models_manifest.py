"""Model artifact manifest for versioned inference."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Resolve relative to repo root so gunicorn/docker cwd does not break loading.
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def manifest_path(ticker: str) -> Path:
    return MODELS_DIR / ticker.upper() / "manifest.json"


def load_manifest(ticker: str) -> dict[str, Any] | None:
    path = manifest_path(ticker)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_manifest(
    ticker: str,
    *,
    model_type: str,
    metrics: dict[str, Any],
    feature_version: str = "1",
    extra: dict[str, Any] | None = None,
) -> Path:
    path = manifest_path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": ticker.upper(),
        "model_type": model_type,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_version": feature_version,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
