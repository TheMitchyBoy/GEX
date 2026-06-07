"""Entry filters for the gamma auto-trader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gex_core.features import safe_float
from gex_core.market_time import export_ts_entry_window_ok, is_entry_window_active
from gex_core.trading.config import (
    block_event_days,
    event_day_size_multiplier,
    max_iv_rank,
    max_strike_distance_pct,
    min_confluence_score,
    min_flow_aggressiveness,
    min_flow_buy_ratio,
    min_gamma_delta,
    min_magnet_progress_pct,
    min_zero_dte_ratio,
    momentum_bars,
    prefer_signal_type,
    require_flow_alignment,
    require_gamma_flip_side,
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


def _strike_distance_pct(spot: float, strike: float) -> float:
    if spot <= 0:
        return 1.0
    return abs(strike - spot) / spot


def _multi_bar_momentum(spots: tuple[float, ...], *, rising: bool) -> bool:
    need = momentum_bars() + 1
    if len(spots) < need:
        return True
    recent = spots[-need:]
    if rising:
        return all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))
    return all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))


def _regime_allows(option_type: str, ctx: MarketContext, strike: float) -> bool:
    regime = (ctx.regime or "").upper()
    spot = ctx.spot
    flip = ctx.gamma_flip
    opt = option_type.lower()
    rising = _multi_bar_momentum(ctx.spot_history, rising=True)
    falling = _multi_bar_momentum(ctx.spot_history, rising=False)

    if require_gamma_flip_side() and flip is not None and flip > 0:
        if opt == "call" and spot < flip:
            return False
        if opt == "put" and spot > flip:
            return False

    if "SHORT" in regime:
        if opt == "call":
            return spot <= strike and rising and (flip is None or spot >= flip)
        return spot >= strike and falling and (flip is None or spot <= flip)

    if opt == "call":
        return spot <= strike and rising
    return spot >= strike and falling


def _magnet_progress(spot: float, strike: float, history: tuple[float, ...]) -> float:
    if len(history) < 2 or spot <= 0 or strike <= 0:
        return 0.0
    start = history[0]
    if start <= 0:
        return 0.0
    if strike >= start:
        denom = max(strike - start, spot * 0.0001)
        return max(0.0, min(1.0, (spot - start) / denom))
    denom = max(start - strike, spot * 0.0001)
    return max(0.0, min(1.0, (start - spot) / denom))


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


def _entry_time_ok(ctx: MarketContext) -> bool:
    if ctx.export_ts:
        return export_ts_entry_window_ok(ctx.export_ts)
    return is_entry_window_active()


def evaluate_entry_filters(
    signals: dict[str, Any],
    *,
    market: MarketContext | None = None,
    uw_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return approve flag, reject reason, and optional size_multiplier."""
    if not strict_entry_filters():
        return {"approve": True, "reason": "Strict filters disabled", "size_multiplier": 1.0}

    rec = signals.get("recommended") or {}
    option_type = str(rec.get("option_type", "call"))
    strike = float(rec.get("strike", 0))
    spot = safe_float(signals.get("spot") or (market.spot if market else 0.0), 0.0)
    gamma_delta = float(rec.get("gamma_delta", 0))
    signal_type = str(rec.get("signal_type", ""))
    selection_reason = str(signals.get("selection_reason", ""))
    size_multiplier = 1.0

    preferred = prefer_signal_type()
    if preferred and signal_type.lower() != preferred:
        return {
            "approve": False,
            "reason": f"Signal type {signal_type} does not match preferred {preferred}",
            "filter": "signal_preference",
            "size_multiplier": 0.0,
        }

    if spot <= 0 or strike <= 0:
        return {"approve": False, "reason": "Missing spot or strike", "filter": "invalid", "size_multiplier": 0.0}

    dist = _strike_distance_pct(spot, strike)
    if dist > max_strike_distance_pct():
        return {
            "approve": False,
            "reason": f"Strike {strike:.0f} is {dist:.1%} from spot (max {max_strike_distance_pct():.1%})",
            "filter": "strike_distance",
            "size_multiplier": 0.0,
        }

    if gamma_delta < min_gamma_delta():
        return {
            "approve": False,
            "reason": f"Gamma delta {gamma_delta:+.3f} below minimum {min_gamma_delta():+.3f}",
            "filter": "gamma_delta",
            "size_multiplier": 0.0,
        }

    if selection_reason == "max_positive_gamma_declined":
        if signal_type != "fastest_gamma_increase":
            return {
                "approve": False,
                "reason": "Max positive gamma declined — only fastest-increase entries allowed via signal layer",
                "filter": "selection_reason",
                "size_multiplier": 0.0,
            }

    market = market or MarketContext(spot=spot)

    if not _entry_time_ok(market):
        return {
            "approve": False,
            "reason": "Outside entry time window (open chop / close decay)",
            "filter": "entry_window",
            "size_multiplier": 0.0,
        }

    min_zdte = min_zero_dte_ratio()
    if min_zdte > 0 and market.zero_dte_ratio is not None and market.zero_dte_ratio < min_zdte:
        return {
            "approve": False,
            "reason": f"0DTE ratio {market.zero_dte_ratio:.2f} below minimum {min_zdte:.2f}",
            "filter": "zero_dte",
            "size_multiplier": 0.0,
        }

    if market.iv_rank is not None and market.iv_rank > max_iv_rank():
        return {
            "approve": False,
            "reason": f"IV rank {market.iv_rank:.2f} above maximum {max_iv_rank():.2f}",
            "filter": "iv_rank",
            "size_multiplier": 0.0,
        }

    min_progress = min_magnet_progress_pct()
    if min_progress > 0:
        progress = _magnet_progress(spot, strike, market.spot_history)
        if progress < min_progress:
            return {
                "approve": False,
                "reason": f"Magnet progress {progress:.1%} below minimum {min_progress:.1%}",
                "filter": "magnet_progress",
                "size_multiplier": 0.0,
            }

    if block_event_days() and (market.is_cpi_day or market.is_nfp_day or market.is_fomc_week):
        event_mult = event_day_size_multiplier()
        if event_mult <= 0:
            return {"approve": False, "reason": "Event day/week — entries blocked", "filter": "event_day", "size_multiplier": 0.0}
        size_multiplier = min(size_multiplier, event_mult)

    if require_spot_momentum() and not _regime_allows(option_type, market, strike):
        return {
            "approve": False,
            "reason": "Spot momentum/regime not aligned with magnet direction",
            "filter": "momentum_regime",
            "size_multiplier": 0.0,
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
            "size_multiplier": 0.0,
        }

    if require_flow_alignment() and not _flow_aligned(option_type, market):
        return {
            "approve": False,
            "reason": "Options flow not aligned with trade direction",
            "filter": "flow",
            "size_multiplier": 0.0,
        }

    return {"approve": True, "reason": "All entry filters passed", "size_multiplier": size_multiplier}
