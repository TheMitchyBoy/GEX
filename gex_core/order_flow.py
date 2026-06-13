"""Options order-flow summary for dashboard panels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gex_core.features import safe_float
from gex_core.predict import load_flow_predictions

TREND_THRESHOLD = 0.15


def _trend_from_score(score: float) -> str:
    if score > TREND_THRESHOLD:
        return "up"
    if score < -TREND_THRESHOLD:
        return "down"
    return "neutral"


def _trend_label(direction: str, event_count: int) -> str:
    if event_count <= 0:
        return "No flow"
    if direction == "up":
        return "Bullish"
    if direction == "down":
        return "Bearish"
    return "Neutral"


def compute_flow_trend(flow: dict[str, Any], *, buy_ratio: float | None = None) -> dict[str, Any]:
    """Derive overall movement direction from flow feed and optional snapshot scalars."""
    event_count = int(flow.get("event_count", 0))
    if event_count <= 0:
        return {
            "direction": "neutral",
            "label": "No flow",
            "score": 0.0,
            "net_delta_gex_bn": 0.0,
            "bullish_strikes": 0,
            "bearish_strikes": 0,
        }

    net_delta = float(flow.get("predicted_flow_delta_gex_bn", 0.0))
    delta_component = max(-1.0, min(1.0, net_delta / 0.05)) if net_delta else 0.0

    signals = flow.get("top_signals") or []
    if signals:
        weighted = sum(float(s.get("score", 0.0)) * abs(float(s.get("score", 0.0))) for s in signals)
        weight_sum = sum(abs(float(s.get("score", 0.0))) for s in signals) or 1.0
        signal_component = weighted / weight_sum
        bullish = sum(1 for s in signals if s.get("direction") == "up")
        bearish = sum(1 for s in signals if s.get("direction") == "down")
    else:
        signal_component = 0.0
        bullish = bearish = 0

    buy_component = 0.0
    if buy_ratio is not None:
        buy_component = max(-1.0, min(1.0, (buy_ratio - 0.5) * 2.0))

    score = 0.5 * signal_component + 0.35 * delta_component + 0.15 * buy_component
    score = max(-1.0, min(1.0, score))
    direction = _trend_from_score(score)

    return {
        "direction": direction,
        "label": _trend_label(direction, event_count),
        "score": round(score, 4),
        "net_delta_gex_bn": round(net_delta, 6),
        "bullish_strikes": bullish,
        "bearish_strikes": bearish,
    }


def build_order_flow_snapshot(
    feed_path: Path,
    *,
    spot: float,
    snapshot: dict[str, Any] | None = None,
    top_n: int = 12,
) -> dict[str, Any]:
    """Load flow feed and return dashboard-ready order-flow payload."""
    snapshot = snapshot or {}
    flow = load_flow_predictions(feed_path, spot=float(spot), top_n=top_n)

    buy_ratio = safe_float(snapshot.get("flow_buy_ratio"))
    aggressiveness = safe_float(snapshot.get("flow_aggressiveness"))
    snapshot_net = safe_float(snapshot.get("flow_net_delta_gex_bn"))

    trend = compute_flow_trend(flow, buy_ratio=buy_ratio)
    if snapshot_net is not None and trend["net_delta_gex_bn"] == 0.0:
        trend = {**trend, "net_delta_gex_bn": round(snapshot_net, 6)}

    strikes = [
        {
            "strike": int(s["strike"]),
            "direction": s.get("direction", "neutral"),
            "score": round(float(s.get("score", 0.0)), 4),
            "flow_imbalance": round(float(s.get("flow_imbalance", 0.0)), 4),
            "recent_gex": round(float(s.get("recent_gex", 0.0)), 6),
            "avg_aggressiveness": round(float(s.get("avg_aggressiveness", 0.0)), 2),
            "flow_gex_bn": round(float(flow.get("flow_by_strike_bn", {}).get(float(s["strike"]), 0.0)), 6),
        }
        for s in flow.get("top_signals", [])
    ]

    return {
        "event_count": flow["event_count"],
        "predicted_flow_delta_gex_bn": round(float(flow.get("predicted_flow_delta_gex_bn", 0.0)), 6),
        "flow_buy_ratio": buy_ratio,
        "flow_aggressiveness": aggressiveness,
        "trend": trend,
        "top_signals": strikes,
    }
