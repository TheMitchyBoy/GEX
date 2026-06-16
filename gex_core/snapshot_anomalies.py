"""Snapshot anomaly detection and optional webhook dispatch."""

from __future__ import annotations

import logging
import os
from typing import Any

from gex_core.features import safe_float

logger = logging.getLogger(__name__)


def _flip_jump_threshold_pct() -> float:
    try:
        return float(os.environ.get("GEX_ANOMALY_FLIP_JUMP_PCT", "0.03"))
    except (TypeError, ValueError):
        return 0.03


def detect_snapshot_anomalies(
    *,
    ticker: str,
    ts: str,
    summary: dict[str, Any],
    features: dict[str, Any],
    prior: dict[str, Any] | None,
    validation: dict[str, Any],
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    spot = safe_float(summary.get("spot"), 0.0)

    if not validation.get("ok", True):
        alerts.append(
            {
                "severity": "high",
                "title": "Snapshot rejected",
                "detail": ", ".join(validation.get("issues") or []),
            }
        )

    if prior and prior.get("strike_count"):
        prior_count = int(prior["strike_count"])
        current_count = int(features.get("strike_count") or 0)
        if prior_count > 0 and current_count < prior_count * 0.5:
            alerts.append(
                {
                    "severity": "medium",
                    "title": "Strike count collapsed",
                    "detail": f"{prior_count} -> {current_count} strikes",
                }
            )

    if prior and spot > 0:
        prior_flip = safe_float(prior.get("gamma_flip"), 0.0)
        flip = safe_float(features.get("gamma_flip"), 0.0)
        if prior_flip > 0 and flip > 0:
            jump = abs(flip - prior_flip) / spot
            if jump >= _flip_jump_threshold_pct():
                alerts.append(
                    {
                        "severity": "medium",
                        "title": "Gamma flip jump",
                        "detail": f"{prior_flip:.0f} -> {flip:.0f} ({jump:.2%} of spot)",
                    }
                )

        prior_regime = str(prior.get("regime") or "")
        regime = str(summary.get("net_gamma_regime") or "")
        if prior_regime and regime and prior_regime != regime:
            alerts.append(
                {
                    "severity": "low",
                    "title": "Regime changed",
                    "detail": f"{prior_regime} -> {regime}",
                }
            )

    quality = safe_float(features.get("quality_score"), 1.0)
    if quality < 0.6:
        alerts.append(
            {
                "severity": "medium",
                "title": "Low data quality score",
                "detail": f"quality_score={quality:.2f}",
            }
        )

    if summary.get("spot_disagreement"):
        alerts.append(
            {
                "severity": "medium",
                "title": "Spot source disagreement",
                "detail": f"{summary.get('spot_disagreement_pct', 0):.3%}",
            }
        )

    return alerts


def maybe_dispatch_quality_alerts(ticker: str, alerts: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not alerts:
        return None
    if os.environ.get("GEX_QUALITY_ALERTS", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    try:
        from gex_core.alert_dispatch import maybe_dispatch_alerts

        return maybe_dispatch_alerts(ticker, alerts, manual=False)
    except Exception:
        logger.debug("quality alert dispatch failed", exc_info=True)
        return None
