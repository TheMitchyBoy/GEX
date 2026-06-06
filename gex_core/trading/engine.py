"""Auto-trader engine — exits, entries, and scheduler integration."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from gex_core.trading.advisor import advise_entry, build_suggestions
from gex_core.trading.config import (
    auto_trader_enabled,
    max_open_positions,
    paper_trading_only,
    stop_loss_pct,
    take_profit_pct,
)
from gex_core.trading.journal import (
    close_trade,
    get_performance_summary,
    is_trader_armed,
    list_open_trades,
    open_trade,
    record_decision,
    set_trader_armed,
)
from gex_core.trading.paper_broker import (
    estimate_entry_premium,
    estimate_option_pnl_pct,
    mark_to_market_premium,
    pnl_usd,
)
from gex_core.trading.signals import compute_gamma_signals

logger = logging.getLogger(__name__)


def trader_status(ticker: str = "SPX") -> dict[str, Any]:
    perf = get_performance_summary(ticker)
    return {
        "enabled": auto_trader_enabled(),
        "armed": is_trader_armed(),
        "paper_mode": paper_trading_only(),
        "stop_loss_pct": stop_loss_pct(),
        "take_profit_pct": take_profit_pct(),
        "max_open_positions": max_open_positions(),
        "open_positions": list_open_trades(ticker),
        "performance": perf,
        "suggestions": build_suggestions(ticker),
    }


def _check_exits(ticker: str, spot: float) -> list[dict[str, Any]]:
    exits: list[dict[str, Any]] = []
    for pos in list_open_trades(ticker):
        pnl_pct = estimate_option_pnl_pct(
            pos["option_type"],
            entry_spot=float(pos["entry_spot"]),
            current_spot=spot,
            strike=float(pos["strike"]),
        )
        exit_reason = None
        if pnl_pct <= -stop_loss_pct():
            exit_reason = "stop_loss"
        elif pnl_pct >= take_profit_pct():
            exit_reason = "take_profit"

        if exit_reason:
            entry_premium = float(pos["entry_premium"])
            exit_premium = mark_to_market_premium(entry_premium, pnl_pct)
            usd = pnl_usd(entry_premium, exit_premium, float(pos.get("qty") or 1))
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
                payload={"trade_id": pos["id"], "pnl_pct": pnl_pct, "reason": exit_reason},
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

    rec = signals["recommended"]
    option_type = advice.get("option_type") or rec["option_type"]
    strike = float(rec["strike"])
    premium = estimate_entry_premium(spot, strike)
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
        meta={"paper": paper_trading_only(), "signals": signals},
    )
    return {
        "action": "opened",
        "trade_id": trade_id,
        "option_type": option_type,
        "strike": strike,
        "premium": premium,
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
    """Run one evaluate cycle: manage exits, maybe open new paper trade."""
    ticker = ticker.upper()
    if not auto_trader_enabled() and not force:
        return {"ran": False, "reason": "Auto-trader disabled (set GEX_AUTO_TRADER=1)"}
    if not is_trader_armed() and not force:
        return {"ran": False, "reason": "Trader disarmed — enable from dashboard"}

    if spot is None or spot <= 0:
        return {"ran": False, "reason": "No spot price"}

    result: dict[str, Any] = {"ran": True, "ticker": ticker, "spot": spot, "exits": [], "entry": None}
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
    set_trader_armed(armed)
