"""Auto-trader engine — exits, entries, and scheduler integration."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

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
    webull_contracts,
    webull_underlying,
)
from gex_core.trading.journal import (
    close_trade,
    get_performance_summary,
    is_trader_armed,
    list_open_trades,
    open_trade,
    patch_trade_meta,
    record_decision,
    set_trader_armed,
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


def _check_exits(ticker: str, spot: float) -> list[dict[str, Any]]:
    broker = get_broker()
    exits: list[dict[str, Any]] = []
    for pos in list_open_trades(ticker):
        pnl_pct = broker.position_pnl_pct(pos, spot=spot)
        if pnl_pct is None:
            continue
        exit_reason = None
        if pnl_pct <= -stop_loss_pct():
            exit_reason = "stop_loss"
        elif pnl_pct >= take_profit_pct():
            exit_reason = "take_profit"

        if not exit_reason:
            continue

        meta = pos.get("meta") or {}
        entry_premium = float(pos["entry_premium"])
        exit_premium = mark_to_market_premium(entry_premium, pnl_pct)
        qty = int(pos.get("qty") or 1)

        if live_trading_allowed():
            sell_limit = max(0.01, entry_premium * (1.0 + pnl_pct) * 0.98)
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
            pnl_pct=pnl_pct,
            pnl_usd=usd,
            exit_reason=exit_reason,
        )
        record_decision(
            ticker=ticker,
            action="exit",
            payload={"trade_id": pos["id"], "pnl_pct": pnl_pct, "reason": exit_reason, "broker": broker.name},
            ai_verdict="closed",
            ai_notes=f"Exit {exit_reason} at {pnl_pct:+.1%}",
        )
        exits.append({"trade_id": pos["id"], "exit_reason": exit_reason, "pnl_pct": pnl_pct})
    return exits


def _maybe_enter(
    *,
    ticker: str,
    spot: float,
    exposure: pd.Series | None,
    previous: pd.Series | None,
    uw_bundle: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if len(list_open_trades(ticker)) >= max_open_positions():
        return None

    signals = compute_gamma_signals(exposure, previous, spot=spot)
    if not signals.get("available"):
        return None

    advice = advise_entry(ticker=ticker, signals=signals, uw_bundle=uw_bundle)
    record_decision(
        ticker=ticker,
        action="entry_review",
        payload={"signals": signals, "advice": advice},
        ai_verdict="approve" if advice.get("approve") else "reject",
        ai_notes=advice.get("reason"),
    )

    if not advice.get("approve"):
        return {"action": "skipped", "reason": advice.get("reason"), "advice": advice}

    broker = get_broker()
    rec = signals["recommended"]
    option_type = advice.get("option_type") or rec["option_type"]
    strike = float(rec["strike"])
    underlying = webull_underlying()
    expire_date = _option_expire_date()
    qty = webull_contracts()

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
        }
    else:
        premium = estimate_entry_premium(spot, strike)
        meta = {"paper": True, "broker": "paper", "underlying": underlying, "expire_date": expire_date, "signals": signals}

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
    )
    result["status"] = trader_status(ticker)
    return result


def arm_trader(armed: bool = True) -> None:
    if armed and live_trading_allowed():
        logger.warning("Arming LIVE Webull auto-trader for %s options", webull_underlying())
    set_trader_armed(armed)
