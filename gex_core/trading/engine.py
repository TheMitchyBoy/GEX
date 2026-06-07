"""Auto-trader engine — exits, entries, and scheduler integration."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from gex_core.features import safe_float
from gex_core.market_time import (
    bars_held_since_entry,
    is_eod_flatten_time,
    is_trader_session_active,
    market_today,
)
from gex_core.trading.advisor import advise_entry, build_suggestions
from gex_core.trading.broker import broker_mode_label, get_broker
from gex_core.trading.config import (
    account_equity_usd,
    auto_trader_enabled,
    eod_flatten_enabled,
    execution_ticker,
    live_trading_allowed,
    max_entries_per_cycle,
    max_open_positions,
    min_entry_confidence,
    paper_trading_only,
    signal_ticker,
    stop_loss_pct,
    take_profit_pct,
    trader_bar_minutes,
    trader_cycle_seconds,
    trader_session_only,
    webull_underlying,
)
from gex_core.trading.execution import execution_summary, map_execution_strike, resolve_execution_spot
from gex_core.trading.exits import ExitProfile, ExitState, build_exit_profile, evaluate_exit
from gex_core.trading.filters import MarketContext, market_context_from_snapshot
from gex_core.trading.journal import (
    close_trade,
    get_account_equity,
    get_account_equity_source,
    get_performance_summary,
    is_trader_armed,
    list_open_trades,
    open_trade,
    patch_trade_meta,
    record_decision,
    reduce_trade_qty,
    set_trader_armed,
    strike_stop_cooldown_active,
)
from gex_core.trading.paper_broker import (
    estimate_entry_premium,
    mark_to_market_premium,
    pnl_usd,
)
from gex_core.trading.signals import compute_entry_candidates
from gex_core.trading.sizing import resolve_contract_qty
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
        "signal_ticker": signal_ticker(),
        "execution_ticker": execution_ticker(),
        "webull_underlying": webull_underlying(),
        "stop_loss_pct": stop_loss_pct(),
        "take_profit_pct": take_profit_pct(),
        "max_open_positions": max_open_positions(),
        "cycle_seconds": trader_cycle_seconds(),
        "bar_minutes": trader_bar_minutes(),
        "session_only": trader_session_only(),
        "account_equity": get_account_equity(),
        "account_equity_source": get_account_equity_source(),
        "starting_equity": account_equity_usd(),
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


def _uses_execution_mapping() -> bool:
    return execution_ticker().upper() != signal_ticker().upper()


def _resolve_execution_context(signal_spot: float) -> tuple[float | None, str]:
    underlying = webull_underlying()
    exec_spot = resolve_execution_spot(signal_spot=signal_spot)
    if _uses_execution_mapping():
        if exec_spot is None or exec_spot <= 0:
            return None, underlying
        return float(exec_spot), underlying
    if exec_spot is None or exec_spot <= 0:
        return float(signal_spot), underlying
    return float(exec_spot), underlying


def _apply_exit(
    *,
    ticker: str,
    pos: dict[str, Any],
    meta: dict[str, Any],
    spot: float,
    exit_reason: str,
    exit_pnl: float,
    sell_qty: int,
    broker: Any,
) -> dict[str, Any] | None:
    entry_premium = float(pos["entry_premium"])
    exit_premium = mark_to_market_premium(entry_premium, exit_pnl)
    qty = int(pos.get("qty") or 1)
    underlying = meta.get("underlying") or webull_underlying()
    expire_date = meta.get("expire_date") or _option_expire_date()
    strike = float(pos["strike"])

    if live_trading_allowed():
        sell_limit = limit_price_for_buy(
            spot,
            strike,
            side="sell",
            underlying=underlying,
            option_type=str(pos["option_type"]),
            expire_date=expire_date,
        )
        sell_result = broker.sell_option(
            underlying=underlying,
            option_type=pos["option_type"],
            strike=strike,
            expire_date=expire_date,
            quantity=sell_qty,
            limit_price=sell_limit,
        )
        if not sell_result.get("ok"):
            logger.error("Webull sell failed for trade %s: %s", pos["id"], sell_result)
            record_decision(
                ticker=ticker,
                action="exit_failed",
                payload={"trade_id": pos["id"], "sell_result": sell_result},
                ai_verdict="error",
            )
            return None
        exit_premium = float(sell_result.get("filled_premium") or sell_limit)

    usd = pnl_usd(entry_premium, exit_premium, sell_qty)
    is_partial = exit_reason == "take_profit_partial" and sell_qty < qty
    if is_partial:
        reduce_trade_qty(int(pos["id"]), qty - sell_qty)
        patch_trade_meta(int(pos["id"]), {"partial_taken": True, "partial_pnl_usd": usd})
        record_decision(
            ticker=ticker,
            action="exit_partial",
            payload={"trade_id": pos["id"], "pnl_pct": exit_pnl, "reason": exit_reason, "sold_qty": sell_qty},
            ai_verdict="partial",
            ai_notes=f"Partial exit {exit_reason} at {exit_pnl:+.1%}",
        )
        return {"trade_id": pos["id"], "exit_reason": exit_reason, "pnl_pct": exit_pnl, "partial": True}

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
    return {"trade_id": pos["id"], "exit_reason": exit_reason, "pnl_pct": exit_pnl}


def _check_exits(ticker: str, spot: float) -> list[dict[str, Any]]:
    broker = get_broker()
    exits: list[dict[str, Any]] = []
    bar_minutes = trader_bar_minutes()
    exec_spot, _ = _resolve_execution_context(signal_spot=spot)
    if _uses_execution_mapping() and (exec_spot is None or exec_spot <= 0):
        return exits
    mark_spot = exec_spot if _uses_execution_mapping() else spot
    for pos in list_open_trades(ticker):
        pnl_pct = broker.position_pnl_pct(pos, spot=mark_spot)
        if pnl_pct is None:
            continue

        meta = pos.get("meta") or {}
        exit_state = _exit_state_from_meta(meta)
        bars_held = bars_held_since_entry(str(pos.get("entry_ts") or ""), bar_minutes=bar_minutes)
        profile = _exit_profile_from_meta(meta, pos)
        magnet_strike = safe_float(pos.get("signal_strike"), 0.0) or None
        exit_reason, exit_pnl = evaluate_exit(
            pnl_pct,
            state=exit_state,
            bars_held=bars_held,
            entry_spot=float(pos["entry_spot"]),
            strike=float(pos["strike"]),
            current_spot=float(mark_spot),
            option_type=str(pos["option_type"]),
            profile=profile,
            magnet_strike=magnet_strike,
        )

        meta_update = {
            "peak_pnl_pct": exit_state.peak_pnl_pct,
            "partial_taken": exit_state.partial_taken,
        }
        patch_trade_meta(int(pos["id"]), meta_update)

        if not exit_reason:
            continue

        qty = int(pos.get("qty") or 1)
        sell_qty = qty
        if exit_reason == "take_profit_partial" and qty > 1:
            sell_qty = max(1, qty // 2)
        elif exit_reason == "take_profit_partial":
            exit_reason = "take_profit"

        result = _apply_exit(
            ticker=ticker,
            pos=pos,
            meta=meta,
            spot=mark_spot,
            exit_reason=exit_reason,
            exit_pnl=exit_pnl,
            sell_qty=sell_qty,
            broker=broker,
        )
        if result:
            exits.append(result)
    return exits


def _flatten_eod(ticker: str, spot: float) -> list[dict[str, Any]]:
    """Close all open 0DTE positions at end-of-day flatten time."""
    if not eod_flatten_enabled() or not is_eod_flatten_time():
        return []
    broker = get_broker()
    exec_spot, _ = _resolve_execution_context(signal_spot=spot)
    if _uses_execution_mapping() and (exec_spot is None or exec_spot <= 0):
        return []
    mark_spot = exec_spot if _uses_execution_mapping() else spot
    closed: list[dict[str, Any]] = []
    for pos in list_open_trades(ticker):
        pnl_pct = broker.position_pnl_pct(pos, spot=mark_spot)
        if pnl_pct is None:
            pnl_pct = 0.0
        result = _apply_exit(
            ticker=ticker,
            pos=pos,
            meta=pos.get("meta") or {},
            spot=mark_spot,
            exit_reason="eod_flatten",
            exit_pnl=pnl_pct,
            sell_qty=int(pos.get("qty") or 1),
            broker=broker,
        )
        if result:
            closed.append(result)
    return closed


def _has_open_duplicate(ticker: str, *, strike: float, option_type: str) -> bool:
    ot = option_type.lower()
    for pos in list_open_trades(ticker):
        if float(pos["strike"]) == strike and str(pos.get("option_type", "")).lower() == ot:
            return True
    return False


def _maybe_enter(
    *,
    ticker: str,
    spot: float,
    exposure: pd.Series | None,
    previous: pd.Series | None,
    uw_bundle: dict[str, Any] | None,
    market: MarketContext | None = None,
) -> dict[str, Any] | None:
    pack = compute_entry_candidates(exposure, previous, spot=spot)
    if not pack.get("available"):
        return None

    broker = get_broker()
    underlying = webull_underlying()
    expire_date = _option_expire_date()
    regime = str((market.regime if market else "") or "")
    exec_spot, _ = _resolve_execution_context(signal_spot=spot)
    if _uses_execution_mapping() and (exec_spot is None or exec_spot <= 0):
        return {"action": "skipped", "reason": "No live SPY spot for SPX sync"}

    last_skip: dict[str, Any] | None = None
    opened_this_cycle = 0
    for rec in pack.get("candidates") or []:
        if len(list_open_trades(ticker)) >= max_open_positions():
            return last_skip
        if opened_this_cycle >= max_entries_per_cycle():
            return last_skip

        signals = {**pack, "recommended": rec}
        advice = advise_entry(ticker=ticker, signals=signals, uw_bundle=uw_bundle, market=market)
        record_decision(
            ticker=ticker,
            action="entry_review",
            payload={"signals": signals, "advice": advice},
            ai_verdict="approve" if advice.get("approve") else "reject",
            ai_notes=advice.get("reason"),
        )

        if not advice.get("approve"):
            last_skip = {"action": "skipped", "reason": advice.get("reason"), "advice": advice}
            continue

        option_type = advice.get("option_type") or rec["option_type"]
        trade_strike = float(rec["strike"])
        magnet_strike = float(rec.get("magnet_strike") or trade_strike)
        if strike_stop_cooldown_active(ticker, magnet_strike, str(option_type)):
            last_skip = {
                "action": "skipped",
                "reason": f"Cooldown active after stop at {magnet_strike:.0f}",
                "advice": advice,
            }
            continue

        exec_strike = (
            map_execution_strike(trade_strike, signal_spot=spot, execution_spot=exec_spot)
            if _uses_execution_mapping()
            else trade_strike
        )
        if _has_open_duplicate(ticker, strike=exec_strike, option_type=str(option_type)):
            last_skip = {"action": "skipped", "reason": f"Already open at {exec_strike:.2f} {option_type}", "advice": advice}
            continue

        confidence = float(advice.get("confidence", 0.5))
        floor = min_entry_confidence()
        if confidence < floor:
            last_skip = {
                "action": "skipped",
                "reason": f"Confidence {confidence:.2f} below minimum {floor:.2f}",
                "advice": advice,
            }
            continue
        size_mult = float(advice.get("size_multiplier") or 1.0)
        exec_map = execution_summary(signal_strike=trade_strike, signal_spot=spot, execution_spot=exec_spot)
        mark_entry_spot = float(exec_spot if _uses_execution_mapping() else spot)
        expected_move = float(market.expected_move_pct) if market and market.expected_move_pct else None
        exit_profile = build_exit_profile(
            ai_confidence=confidence,
            gamma_delta=float(rec["gamma_delta"]),
            regime=regime,
            entry_spot=mark_entry_spot,
            strike=float(exec_strike),
            expected_move_pct=expected_move,
        )
        profile_meta = {
            "hold_for_target": exit_profile.hold_for_target,
            "partial_take_profit": exit_profile.partial_take_profit,
            "trail_trigger": exit_profile.trail_trigger,
            "trail_floor": exit_profile.trail_floor,
            "time_stop_bars": exit_profile.time_stop_bars,
            "full_take_profit": exit_profile.full_take_profit,
        }

        premium_est = estimate_entry_premium(mark_entry_spot, exec_strike)
        qty = resolve_contract_qty(
            confidence=confidence,
            premium=premium_est,
            entry_spot=mark_entry_spot,
            strike=float(exec_strike),
            account_equity=get_account_equity(),
            size_multiplier=size_mult,
        )
        if qty < 1:
            last_skip = {
                "action": "skipped",
                "reason": "Risk sizing blocked entry (insufficient budget)",
                "advice": advice,
            }
            continue

        if live_trading_allowed():
            limit_price = limit_price_for_buy(
                exec_spot,
                exec_strike,
                side="buy",
                underlying=underlying,
                option_type=str(option_type),
                expire_date=expire_date,
            )
            order = broker.buy_option(
                underlying=underlying,
                option_type=option_type,
                strike=exec_strike,
                expire_date=expire_date,
                quantity=qty,
                limit_price=limit_price,
                spot=exec_spot,
            )
            if not order.get("ok"):
                reason = order.get("stage") or "Webull order rejected"
                last_skip = {"action": "order_failed", "reason": reason, "order": order, "advice": advice}
                continue
            filled_qty = int(order.get("filled_qty") or qty)
            if filled_qty <= 0:
                last_skip = {"action": "order_failed", "reason": "No fill received", "order": order, "advice": advice}
                continue
            qty = filled_qty
            premium = float(order.get("filled_premium") or order.get("limit_price") or limit_price)
            meta = {
                "paper": False,
                "broker": "webull",
                "underlying": underlying,
                "expire_date": expire_date,
                "webull_client_order_id": order.get("client_order_id"),
                "execution_map": exec_map,
                "signals": signals,
                "order": {"preview": order.get("preview"), "response": order.get("response"), "fill": order.get("fill_detail")},
                "peak_pnl_pct": 0.0,
                "partial_taken": False,
                "regime": regime,
                "exit_profile": profile_meta,
            }
        else:
            premium = estimate_entry_premium(exec_spot if _uses_execution_mapping() else spot, exec_strike)
            meta = {
                "paper": True,
                "broker": "paper",
                "underlying": underlying,
                "expire_date": expire_date,
                "execution_map": exec_map if _uses_execution_mapping() else None,
                "signals": signals,
                "peak_pnl_pct": 0.0,
                "partial_taken": False,
                "regime": regime,
                "exit_profile": profile_meta,
            }

        trade_id = open_trade(
            ticker=ticker,
            option_type=option_type,
            strike=exec_strike,
            entry_spot=exec_spot if _uses_execution_mapping() else spot,
            entry_premium=premium,
            signal_type=rec["signal_type"],
            signal_strike=magnet_strike,
            signal_gamma=float(rec["gamma_bn"]),
            gamma_delta=float(rec["gamma_delta"]),
            ai_confidence=float(advice.get("confidence", 0.5)),
            ai_reason=str(advice.get("reason", "")),
            meta=meta,
            qty=float(qty),
        )
        if live_trading_allowed() and meta.get("webull_client_order_id"):
            patch_trade_meta(trade_id, {"webull_client_order_id": meta["webull_client_order_id"]})

        opened_this_cycle += 1
        return {
            "action": "opened",
            "trade_id": trade_id,
            "option_type": option_type,
            "strike": exec_strike,
            "signal_strike": magnet_strike,
            "trade_strike": trade_strike,
            "execution_map": exec_map,
            "premium": premium,
            "broker": broker.name,
            "advice": advice,
        }

    return last_skip


def run_trading_cycle(
    *,
    ticker: str = "SPX",
    spot: float | None = None,
    exposure: pd.Series | None = None,
    previous_exposure: pd.Series | None = None,
    uw_bundle: dict[str, Any] | None = None,
    snapshot: dict[str, Any] | None = None,
    previous_spot: float | None = None,
    spot_history: list[float] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run one evaluate cycle: manage exits, maybe open new trade."""
    ticker = ticker.upper()
    if not auto_trader_enabled() and not force:
        return {"ran": False, "reason": "Auto-trader disabled (set GEX_AUTO_TRADER=1)"}
    if not force and trader_session_only() and not is_trader_session_active():
        return {"ran": False, "reason": "Outside market session"}
    if not is_trader_armed() and not force:
        return {"ran": False, "reason": "Trader disarmed — enable from dashboard"}
    if live_trading_allowed() and not webull_underlying():
        return {"ran": False, "reason": "Webull underlying symbol not configured"}

    if spot is None or spot <= 0:
        return {"ran": False, "reason": "No spot price"}

    snap = dict(snapshot or {})
    snap.setdefault("spot", spot)
    if snap.get("ts") is None and snap.get("export_ts"):
        snap["ts"] = snap["export_ts"]
    history = list(spot_history or [])
    if not history and previous_spot is not None:
        history = [float(previous_spot), float(spot)]
    elif not history:
        history = [float(spot)]
    market = market_context_from_snapshot(snap, prev_spot=previous_spot, spot_history=history)

    result: dict[str, Any] = {
        "ran": True,
        "ticker": ticker,
        "spot": spot,
        "broker_mode": broker_mode_label(),
        "exits": [],
        "eod_exits": [],
        "entry": None,
    }
    result["eod_exits"] = _flatten_eod(ticker, float(spot))
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
