"""Entry filters for the gamma auto-trader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gex_core.features import safe_float
from gex_core.trading.config import (
    block_event_days,
    max_strike_distance_pct,
    min_confluence_score,
    min_gamma_delta,
    require_flow_alignment,
    require_spot_momentum,
    strict_entry_filters,
)


@dataclass(frozen=True)
class MarketContext:
    spot: float
    prev_spot: float | None = None
    gamma_flip: float | None = None
    regime: str | None = None
    is_cpi_day: bool = False
    is_nfp_day: bool = False
    is_fomc_week: bool = False
    flow_net_delta_gex_bn: float | None = None
    confluence_score: float | None = None


def market_context_from_snapshot(snapshot: dict[str, Any] | None, *, prev_spot: float | None = None) -> MarketContext:
    snap = snapshot or {}
    return MarketContext(
        spot=safe_float(snap.get("spot"), 0.0),
        prev_spot=prev_spot,
        gamma_flip=safe_float(snap.get("gamma_flip"), 0.0) or None,
        regime=str(snap.get("regime") or ""),
        is_cpi_day=bool(snap.get("is_cpi_day")),
        is_nfp_day=bool(snap.get("is_nfp_day")),
        is_fomc_week=bool(snap.get("is_fomc_week")),
        flow_net_delta_gex_bn=safe_float(snap.get("flow_net_delta_gex_bn"), 0.0) or None,
        confluence_score=safe_float(snap.get("confluence_score"), 0.0) or None,
    )


def _strike_distance_pct(spot: float, strike: float) -> float:
    if spot <= 0:
        return 1.0
    return abs(strike - spot) / spot


def _regime_allows(option_type: str, ctx: MarketContext, strike: float) -> bool:
    regime = (ctx.regime or "").upper()
    prev = ctx.prev_spot
    spot = ctx.spot
    flip = ctx.gamma_flip
    opt = option_type.lower()

    if prev is None or prev <= 0:
        return True

    rising = spot > prev
    falling = spot < prev

    if "SHORT" in regime:
        if opt == "call":
            return spot <= strike and rising and (flip is None or spot >= flip)
        return spot >= strike and falling and (flip is None or spot <= flip)

    if opt == "call":
        return spot <= strike and rising
    return spot >= strike and falling


def _flow_aligned(option_type: str, ctx: MarketContext) -> bool:
    flow = ctx.flow_net_delta_gex_bn
    if flow is None or abs(flow) < 0.01:
        return True
    if option_type.lower() == "call":
        return flow >= 0
    return flow <= 0


def evaluate_entry_filters(
    signals: dict[str, Any],
    *,
    market: MarketContext | None = None,
    uw_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return approve flag and reject reason for a candidate signal."""
    if not strict_entry_filters():
        return {"approve": True, "reason": "Strict filters disabled"}

    rec = signals.get("recommended") or {}
    option_type = str(rec.get("option_type", "call"))
    strike = float(rec.get("strike", 0))
    spot = safe_float(signals.get("spot") or (market.spot if market else 0.0), 0.0)
    gamma_delta = float(rec.get("gamma_delta", 0))
    selection_reason = str(signals.get("selection_reason", ""))

    if spot <= 0 or strike <= 0:
        return {"approve": False, "reason": "Missing spot or strike", "filter": "invalid"}

    dist = _strike_distance_pct(spot, strike)
    if dist > max_strike_distance_pct():
        return {
            "approve": False,
            "reason": f"Strike {strike:.0f} is {dist:.1%} from spot (max {max_strike_distance_pct():.1%})",
            "filter": "strike_distance",
        }

    if gamma_delta < min_gamma_delta():
        return {
            "approve": False,
            "reason": f"Gamma delta {gamma_delta:+.3f} below minimum {min_gamma_delta():+.3f}",
            "filter": "gamma_delta",
        }

    if selection_reason == "max_positive_gamma_declined":
        if str(rec.get("signal_type", "")) != "fastest_gamma_increase":
            return {
                "approve": False,
                "reason": "Max positive gamma declined — only fastest-increase entries allowed via signal layer",
                "filter": "selection_reason",
            }

    market = market or MarketContext(spot=spot)
    if block_event_days() and (market.is_cpi_day or market.is_nfp_day or market.is_fomc_week):
        return {"approve": False, "reason": "Event day/week — entries blocked", "filter": "event_day"}

    if require_spot_momentum() and not _regime_allows(option_type, market, strike):
        return {
            "approve": False,
            "reason": "Spot momentum/regime not aligned with magnet direction",
            "filter": "momentum_regime",
        }

    confluence = market.confluence_score
    if uw_bundle:
        summary = uw_bundle.get("summary") or {}
        confluence = confluence or safe_float(summary.get("confluence_score"), 0.0) or None
    if confluence is not None and confluence < min_confluence_score():
        return {
            "approve": False,
            "reason": f"Confluence {confluence:.0f} below minimum {min_confluence_score():.0f}",
            "filter": "confluence",
        }

    if require_flow_alignment() and not _flow_aligned(option_type, market):
        return {
            "approve": False,
            "reason": "Options flow not aligned with trade direction",
            "filter": "flow",
        }

    return {"approve": True, "reason": "All entry filters passed"}
