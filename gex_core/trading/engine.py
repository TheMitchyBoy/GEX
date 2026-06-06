"""Auto-trader engine — exits, entries, and scheduler integration."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from gex_core.features import safe_float
from gex_core.market_time import market_today
from gex_core.trading.advisor import advise_entry, build_suggestions
from gex_core.trading.broker import broker_mode_label, get_broker
from gex_core.trading.config import (
    auto_trader_enabled,
    live_trading_allowed,
    max_open_positions,
    paper_trading_only,
    stop_loss_pct,
    take_profit_pct,
    webull_underlying,
)
from gex_core.trading.exits import ExitProfile, ExitState, build_exit_profile, contracts_for_confidence, evaluate_exit
from gex_core.trading.filters import MarketContext, market_context_from_snapshot
from gex_core.trading.journal import (
    close_trade,
    get_performance_summary,
    is_trader_armed,
    list_open_trades,
    open_trade,
    patch_trade_meta,
    record_decision,
    set_trader_armed,
    strike_stop_cooldown_active,
)
from gex_core.trading.paper_broker import (
    estimate_entry_premium,
    mark_to_market_premium,
    pnl_usd,
)
from gex_core.trading.signals import compute_gamma_signals
from gex_core.trading.webull_broker import limit_price_for_buy

logger = logging.getLogger(__name__)


def trader_status(ticker: str = "SPX") -> dict[str, Any]:
    perf = get_performance_summary(ticker)
    mode = broker_mode_label()
    return {
        "enabled": auto_trader_enabled(),
        "armed": is_trader_armed(),
        "paper_mode": paper_trading_only(),
        "live_mode": live_trading_allowed(),
        "broker_mode": mode,
        "webull_underlying": webull_underlying(),
        "stop_loss_pct": stop_loss_pct(),
        "take_profit_pct": take_profit_pct(),
        "max_open_positions": max_open_positions(),
        "open_positions": list_open_trades(ticker),
        "performance": perf,
        "suggestions": build_suggestions(ticker),
    }


def _option_expire_date() -> str:
    return market_today()


def _exit_state_from_meta(meta: dict[str, Any]) -> ExitState:
    return ExitState(
        peak_pnl_pct=float(meta.get("peak_pnl_pct") or 0.0),
        partial_taken=bool(meta.get("partial_taken")),
    )


def _exit_profile_from_meta(meta: dict[str, Any], pos: dict[str, Any]) -> ExitProfile:
    saved = meta.get("exit_profile") or {}
    if saved:
        return ExitProfile(
            hold_for_target=bool(saved.get("hold_for_target")),
            partial_take_profit=saved.get("partial_take_profit"),
            trail_trigger=float(saved.get("trail_trigger", 0.10)),
            trail_floor=float(saved.get("trail_floor", 0.05)),
            time_stop_bars=int(saved.get("time_stop_bars", 6)),
            full_take_profit=float(saved.get("full_take_profit", take_profit_pct())),
        )
    return build_exit_profile(
        ai_confidence=float(pos.get("ai_confidence") or 0.5),
        gamma_delta=float(pos.get("gamma_delta") or 0.0),
        regime=str(meta.get("regime") or ""),
        entry_spot=float(pos["entry_spot"]),
        strike=float(pos["strike"]),
    )


def _check_exits(ticker: str, spot: float, *, bar_count: int = 0) -> list[dict[str, Any]]:
    broker = get_broker()
    exits: list[dict[str, Any]] = []
    for pos in list_open_trades(ticker):
        pnl_pct = broker.position_pnl_pct(pos, spot=spot)
        if pnl_pct is None:
            continue

        meta = pos.get("meta") or {}
        exit_state = _exit_state_from_meta(meta)
        bars_held = int(meta.get("bars_held") or bar_count or 0)
        profile = _exit_profile_from_meta(meta, pos)
        exit_reason, exit_pnl = evaluate_exit(
            pnl_pct,
            state=exit_state,
            bars_held=bars_held,
            entry_spot=float(pos["entry_spot"]),
            strike=float(pos["strike"]),
            current_spot=float(spot),
            option_type=str(pos["option_type"]),
            profile=profile,
        )

        meta_update = {
            "peak_pnl_pct": exit_state.peak_pnl_pct,
            "partial_taken": exit_state.partial_taken,
            "bars_held": bars_held + 1,
        }
        patch_trade_meta(int(pos["id"]), meta_update)

        if not exit_reason:
            continue

        entry_premium = float(pos["entry_premium"])
        exit_premium = mark_to_market_premium(entry_premium, exit_pnl)
        qty = int(pos.get("qty") or 1)

        if live_trading_allowed():
            sell_limit = max(0.01, entry_premium * (1.0 + exit_pnl) * 0.98)
            sell_result = broker.sell_option(
                underlying=meta.get("underlying") or webull_underlying(),
                option_type=pos["option_type"],
                strike=float(pos["strike"]),
                expire_date=meta.get("expire_date") or _option_expire_date(),
                quantity=qty,
                limit_price=sell_limit,
                client_order_id=meta.get("webull_client_order_id"),
            )
            if not sell_result.get("ok"):
                logger.error("Webull sell failed for trade %s: %s", pos["id"], sell_result)
                record_decision(
                    ticker=ticker,
                    action="exit_failed",
                    payload={"trade_id": pos["id"], "sell_result": sell_result},
                    ai_verdict="error",
                )
                continue
            exit_premium = sell_limit

        usd = pnl_usd(entry_premium, exit_premium, qty)
        close_trade(
            int(pos["id"]),
            exit_spot=spot,
            exit_premium=exit_premium,
            pnl_pct=exit_pnl,
            pnl_usd=usd,
            exit_reason=exit_reason,
        )
        record_decision(
            ticker=ticker,
            action="exit",
            payload={"trade_id": pos["id"], "pnl_pct": exit_pnl, "reason": exit_reason, "broker": broker.name},
            ai_verdict="closed",
            ai_notes=f"Exit {exit_reason} at {exit_pnl:+.1%}",
        )
        exits.append({"trade_id": pos["id"], "exit_reason": exit_reason, "pnl_pct": exit_pnl})
    return exits


def _maybe_enter(
    *,
    ticker: str,
    spot: float,
    exposure: pd.Series | None,
    previous: pd.Series | None,
    uw_bundle: dict[str, Any] | None,
    market: MarketContext | None = None,
) -> dict[str, Any] | None:
    if len(list_open_trades(ticker)) >= max_open_positions():
        return None

    signals = compute_gamma_signals(exposure, previous, spot=spot)
    if not signals.get("available"):
        return None

    advice = advise_entry(ticker=ticker, signals=signals, uw_bundle=uw_bundle, market=market)
    record_decision(
        ticker=ticker,
        action="entry_review",
        payload={"signals": signals, "advice": advice},
        ai_verdict="approve" if advice.get("approve") else "reject",
        ai_notes=advice.get("reason"),
    )

    if not advice.get("approve"):
        return {"action": "skipped", "reason": advice.get("reason"), "advice": advice}

    rec = signals["recommended"]
    option_type = advice.get("option_type") or rec["option_type"]
    strike = float(rec["strike"])
    if strike_stop_cooldown_active(ticker, strike, str(option_type)):
        return {"action": "skipped", "reason": f"Cooldown active after stop at {strike:.0f}", "advice": advice}

    broker = get_broker()
    confidence = float(advice.get("confidence", 0.5))
    qty = int(contracts_for_confidence(confidence))
    underlying = webull_underlying()
    expire_date = _option_expire_date()
    regime = str((market.regime if market else "") or "")
    exit_profile = build_exit_profile(
        ai_confidence=confidence,
        gamma_delta=float(rec["gamma_delta"]),
        regime=regime,
        entry_spot=float(spot),
        strike=strike,
    )
    profile_meta = {
        "hold_for_target": exit_profile.hold_for_target,
        "partial_take_profit": exit_profile.partial_take_profit,
        "trail_trigger": exit_profile.trail_trigger,
        "trail_floor": exit_profile.trail_floor,
        "time_stop_bars": exit_profile.time_stop_bars,
        "full_take_profit": exit_profile.full_take_profit,
    }

    if live_trading_allowed():
        limit_price = limit_price_for_buy(spot, strike, side="buy")
        order = broker.buy_option(
            underlying=underlying,
            option_type=option_type,
            strike=strike,
            expire_date=expire_date,
            quantity=qty,
            limit_price=limit_price,
            spot=spot,
        )
        if not order.get("ok"):
            return {"action": "order_failed", "reason": "Webull order rejected", "order": order, "advice": advice}
        premium = float(order.get("filled_premium") or order.get("limit_price") or limit_price)
        meta = {
            "paper": False,
            "broker": "webull",
            "underlying": underlying,
            "expire_date": expire_date,
            "webull_client_order_id": order.get("client_order_id"),
            "signals": signals,
            "order": {"preview": order.get("preview"), "response": order.get("response")},
            "peak_pnl_pct": 0.0,
            "partial_taken": False,
            "bars_held": 0,
            "regime": regime,
            "exit_profile": profile_meta,
        }
    else:
        premium = estimate_entry_premium(spot, strike)
        meta = {
            "paper": True,
            "broker": "paper",
            "underlying": underlying,
            "expire_date": expire_date,
            "signals": signals,
            "peak_pnl_pct": 0.0,
            "partial_taken": False,
            "bars_held": 0,
            "regime": regime,
            "exit_profile": profile_meta,
        }

    trade_id = open_trade(
        ticker=ticker,
        option_type=option_type,
        strike=strike,
        entry_spot=spot,
        entry_premium=premium,
        signal_type=rec["signal_type"],
        signal_strike=strike,
        signal_gamma=float(rec["gamma_bn"]),
        gamma_delta=float(rec["gamma_delta"]),
        ai_confidence=float(advice.get("confidence", 0.5)),
        ai_reason=str(advice.get("reason", "")),
        meta=meta,
        qty=float(qty),
    )
    if live_trading_allowed() and meta.get("webull_client_order_id"):
        patch_trade_meta(trade_id, {"webull_client_order_id": meta["webull_client_order_id"]})

    return {
        "action": "opened",
        "trade_id": trade_id,
        "option_type": option_type,
        "strike": strike,
        "premium": premium,
        "broker": broker.name,
        "advice": advice,
    }


def run_trading_cycle(
    *,
    ticker: str = "SPX",
    spot: float | None = None,
    exposure: pd.Series | None = None,
    previous_exposure: pd.Series | None = None,
    uw_bundle: dict[str, Any] | None = None,
    snapshot: dict[str, Any] | None = None,
    previous_spot: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run one evaluate cycle: manage exits, maybe open new trade."""
    ticker = ticker.upper()
    if not auto_trader_enabled() and not force:
        return {"ran": False, "reason": "Auto-trader disabled (set GEX_AUTO_TRADER=1)"}
    if not is_trader_armed() and not force:
        return {"ran": False, "reason": "Trader disarmed — enable from dashboard"}
    if live_trading_allowed() and not webull_underlying():
        return {"ran": False, "reason": "Webull underlying symbol not configured"}

    if spot is None or spot <= 0:
        return {"ran": False, "reason": "No spot price"}

    snap = dict(snapshot or {})
    snap.setdefault("spot", spot)
    market = market_context_from_snapshot(snap, prev_spot=previous_spot)

    result: dict[str, Any] = {
        "ran": True,
        "ticker": ticker,
        "spot": spot,
        "broker_mode": broker_mode_label(),
        "exits": [],
        "entry": None,
    }
    result["exits"] = _check_exits(ticker, float(spot))
    result["entry"] = _maybe_enter(
        ticker=ticker,
        spot=float(spot),
        exposure=exposure,
        previous=previous_exposure,
        uw_bundle=uw_bundle,
        market=market,
    )
    result["status"] = trader_status(ticker)
    return result


def arm_trader(armed: bool = True) -> None:
    if armed and live_trading_allowed():
        logger.warning("Arming LIVE Webull auto-trader for %s options", webull_underlying())
    set_trader_armed(armed)
