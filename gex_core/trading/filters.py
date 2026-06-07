"""Entry filters for the auto-trader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gex_core.features import safe_float
from gex_core.market_time import MARKET_TZ, parse_export_ts_utc
from gex_core.trading.config import (
    entry_time_filter_enabled,
    entry_window_after_open_min,
    entry_window_before_close_min,
    min_flow_aggressiveness,
    min_flow_buy_ratio,
    min_gamma_delta,
    min_magnet_distance_pct,
    min_magnet_progress_pct,
    momentum_bars,
    regime_strict,
    require_flow_alignment,
    require_gamma_flip_side,
    require_spot_momentum,
    strict_entry_filters,
)
from gex_core.trading.exits import spot_progress_toward_strike


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


def _spot_momentum_aligned(option_type: str, ctx: MarketContext) -> bool:
    if not require_spot_momentum():
        return True
    history = ctx.spot_history
    bars = momentum_bars()
    if len(history) < bars + 1:
        return True
    start = float(history[-(bars + 1)])
    end = float(history[-1])
    if option_type.lower() == "call":
        return end > start
    return end < start


def _gamma_flip_aligned(option_type: str, ctx: MarketContext) -> bool:
    if not require_gamma_flip_side():
        return True
    flip = ctx.gamma_flip
    if flip is None or flip <= 0:
        return True
    spot = ctx.spot
    if option_type.lower() == "call":
        return spot > flip
    return spot < flip


def _entry_time_ok(ctx: MarketContext) -> bool:
    if not entry_time_filter_enabled() or not ctx.export_ts:
        return True
    try:
        local = parse_export_ts_utc(ctx.export_ts).astimezone(MARKET_TZ)
    except (TypeError, ValueError):
        return True
    minutes = local.hour * 60 + local.minute
    open_min = 9 * 60 + 30 + entry_window_after_open_min()
    close_min = 16 * 60 - entry_window_before_close_min()
    return open_min <= minutes <= close_min


def _regime_ok(ctx: MarketContext) -> bool:
    if not regime_strict():
        return True
    return "SHORT" not in (ctx.regime or "").upper()


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
    magnet_strike = float(rec.get("magnet_strike") or 0)

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

    master = signals.get("master_direction")
    if master and str(master).lower() != option_type.lower():
        return {
            "approve": False,
            "reason": f"Trade direction {option_type} conflicts with max-gamma direction {master}",
            "filter": "direction_lock",
            "size_multiplier": 0.0,
        }

    min_dist = min_magnet_distance_pct()
    if min_dist > 0 and magnet_strike > 0:
        dist = abs(magnet_strike - spot) / spot
        if dist < min_dist:
            return {
                "approve": False,
                "reason": f"Magnet too close to spot ({dist:.2%} < {min_dist:.2%})",
                "filter": "magnet_distance",
                "size_multiplier": 0.0,
            }

    min_prog = min_magnet_progress_pct()
    if min_prog > 0 and magnet_strike > 0:
        anchor = float(market.prev_spot or spot)
        progress = spot_progress_toward_strike(
            entry_spot=anchor,
            current_spot=spot,
            strike=magnet_strike,
            option_type=option_type,
        )
        if progress < min_prog:
            return {
                "approve": False,
                "reason": f"Spot progress toward magnet {progress:.0%} below {min_prog:.0%}",
                "filter": "magnet_progress",
                "size_multiplier": 0.0,
            }

    if not _regime_ok(market):
        return {
            "approve": False,
            "reason": "Short-gamma regime — entries blocked",
            "filter": "regime",
            "size_multiplier": 0.0,
        }

    if not _entry_time_ok(market):
        return {
            "approve": False,
            "reason": "Outside allowed entry time window",
            "filter": "entry_time",
            "size_multiplier": 0.0,
        }

    if not _spot_momentum_aligned(option_type, market):
        return {
            "approve": False,
            "reason": "Spot not moving in trade direction over recent bars",
            "filter": "momentum",
            "size_multiplier": 0.0,
        }

    if not _gamma_flip_aligned(option_type, market):
        return {
            "approve": False,
            "reason": "Spot on wrong side of gamma flip for trade direction",
            "filter": "gamma_flip",
            "size_multiplier": 0.0,
        }

    if require_flow_alignment() and not _flow_aligned(option_type, market):
        return {
            "approve": False,
            "reason": "Options flow not aligned with trade direction",
            "filter": "flow",
            "size_multiplier": 0.0,
        }

    return {"approve": True, "reason": "Entry filters passed", "size_multiplier": 1.0}
