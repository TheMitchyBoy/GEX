"""Entry filters for the auto-trader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gex_core.features import safe_float
from gex_core.trading.config import (
    min_flow_aggressiveness,
    min_flow_buy_ratio,
    min_gamma_delta,
    require_flow_alignment,
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
    zero_dte_ratio: float | None = None
    iv_rank: float | None = None
    expected_move_pct: float | None = None
    flow_buy_ratio: float | None = None
    flow_aggressiveness: float | None = None
    spot_history: tuple[float, ...] = field(default_factory=tuple)
    export_ts: str | None = None


def market_context_from_snapshot(
    snapshot: dict[str, Any] | None,
    *,
    prev_spot: float | None = None,
    spot_history: list[float] | None = None,
) -> MarketContext:
    snap = snapshot or {}
    history = tuple(spot_history or [])
    if not history and prev_spot is not None:
        history = (float(prev_spot), float(safe_float(snap.get("spot"), 0.0)))
    elif not history:
        spot = safe_float(snap.get("spot"), 0.0)
        history = (spot,) if spot > 0 else tuple()

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
        zero_dte_ratio=safe_float(snap.get("zero_dte_ratio"), 0.0) or None,
        iv_rank=safe_float(snap.get("iv_rank"), 0.0) or None,
        expected_move_pct=safe_float(snap.get("expected_move_pct"), 0.0) or None,
        flow_buy_ratio=safe_float(snap.get("flow_buy_ratio"), 0.0) or None,
        flow_aggressiveness=safe_float(snap.get("flow_aggressiveness"), 0.0) or None,
        spot_history=history,
        export_ts=str(snap.get("ts") or "") or None,
    )


def _flow_aligned(option_type: str, ctx: MarketContext) -> bool:
    flow = ctx.flow_net_delta_gex_bn
    buy_ratio = ctx.flow_buy_ratio
    aggressiveness = ctx.flow_aggressiveness
    opt = option_type.lower()

    if buy_ratio is not None and buy_ratio > 0:
        if opt == "call" and buy_ratio < min_flow_buy_ratio():
            return False
        if opt == "put" and buy_ratio > (1.0 - min_flow_buy_ratio()):
            return False

    min_agg = min_flow_aggressiveness()
    if min_agg > 0 and aggressiveness is not None and aggressiveness < min_agg:
        return False

    if flow is None or abs(flow) < 0.01:
        return True
    if opt == "call":
        return flow >= 0
    return flow <= 0


def evaluate_entry_filters(
    signals: dict[str, Any],
    *,
    market: MarketContext | None = None,
    uw_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return approve flag, reject reason, and optional size_multiplier."""
    _ = uw_bundle
    if not strict_entry_filters():
        return {"approve": True, "reason": "Strict filters disabled", "size_multiplier": 1.0}

    rec = signals.get("recommended") or {}
    option_type = str(rec.get("option_type", "call"))
    strike = float(rec.get("strike", 0))
    spot = safe_float(signals.get("spot") or (market.spot if market else 0.0), 0.0)
    gamma_delta = float(rec.get("gamma_delta", 0))

    if spot <= 0 or strike <= 0:
        return {"approve": False, "reason": "Missing spot or strike", "filter": "invalid", "size_multiplier": 0.0}

    if min_gamma_delta() > 0 and gamma_delta < min_gamma_delta():
        return {
            "approve": False,
            "reason": f"Gamma delta {gamma_delta:+.3f} below minimum {min_gamma_delta():+.3f}",
            "filter": "gamma_delta",
            "size_multiplier": 0.0,
        }

    market = market or MarketContext(spot=spot)

    if require_flow_alignment() and not _flow_aligned(option_type, market):
        return {
            "approve": False,
            "reason": "Options flow not aligned with trade direction",
            "filter": "flow",
            "size_multiplier": 0.0,
        }

    return {"approve": True, "reason": "Entry filters passed (gamma delta and flow)", "size_multiplier": 1.0}
