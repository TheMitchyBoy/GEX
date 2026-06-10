"""Execute one low-GEX-direction trade cycle (signal → paper or Webull order)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from gex_core.market_time import is_trader_session_active, market_today
from gex_core.trading.broker import broker_mode_label, get_broker
from gex_core.trading.config import (
    execution_ticker,
    live_trading_allowed,
    low_gex_reenter_each_bar,
    max_open_positions,
    paper_trading_only,
    signal_ticker,
    webull_underlying,
)
from gex_core.trading.execution import execution_summary, map_execution_strike, resolve_execution_spot
from gex_core.trading.journal import (
    close_trade,
    get_account_equity,
    list_open_trades,
    open_trade,
    patch_trade_meta,
    record_decision,
)
from gex_core.trading.low_gex_signals import compute_low_gex_signal
from gex_core.trading.paper_broker import estimate_entry_premium, mark_to_market_premium, pnl_usd
from gex_core.trading.sizing import resolve_contract_qty
from gex_core.trading.webull_broker import limit_price_for_buy

logger = logging.getLogger(__name__)


def _uses_execution_mapping() -> bool:
    return execution_ticker().upper() != signal_ticker().upper()


def _flatten_open_positions(ticker: str, *, spot: float, reason: str = "bar_rotation") -> list[dict[str, Any]]:
    """Close every open position at the current mark (5-min rotation mode)."""
    from gex_core.trading.execution import resolve_execution_spot

    broker = get_broker()
    exec_spot = float(spot)
    if _uses_execution_mapping():
        mapped = resolve_execution_spot(signal_spot=spot)
        if mapped is None or mapped <= 0:
            return []
        exec_spot = float(mapped)

    closed: list[dict[str, Any]] = []
    for pos in list_open_trades(ticker):
        pnl_pct = broker.position_pnl_pct(pos, spot=exec_spot)
        if pnl_pct is None:
            pnl_pct = 0.0
        qty = int(pos.get("qty") or 1)
        entry_premium = float(pos["entry_premium"])
        exit_premium = mark_to_market_premium(entry_premium, pnl_pct)
        close_trade(
            int(pos["id"]),
            exit_spot=exec_spot,
            exit_premium=exit_premium,
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd(entry_premium, exit_premium, qty),
            exit_reason=reason,
        )
        closed.append({"trade_id": pos["id"], "pnl_pct": pnl_pct, "reason": reason})
    return closed


def _has_open_duplicate(ticker: str, *, strike: float, option_type: str) -> bool:
    ot = option_type.lower()
    for pos in list_open_trades(ticker):
        if float(pos["strike"]) == strike and str(pos.get("option_type", "")).lower() == ot:
            return True
    return False


def fetch_gex_exposure(ticker: str) -> tuple[float, pd.Series]:
    """Load spot and net gamma-by-strike from UW."""
    from gex_core.data_source import fetch_gex_data

    result = fetch_gex_data(ticker)
    agg = result.aggregates
    return float(result.spot), pd.Series(agg.gex_by_strike, dtype=float).sort_index()


def run_low_gex_trade(
    *,
    ticker: str = "SPX",
    spot: float | None = None,
    exposure: pd.Series | None = None,
    execute: bool = False,
    session_check: bool = True,
    reenter_each_bar: bool | None = None,
) -> dict[str, Any]:
    """Evaluate lowest-GEX signal and optionally open a position."""
    ticker = ticker.upper()
    out: dict[str, Any] = {
        "ticker": ticker,
        "broker_mode": broker_mode_label(),
        "executed": False,
    }

    if session_check and not is_trader_session_active():
        out["ran"] = False
        out["reason"] = "Outside market session"
        return out

    if spot is None or exposure is None:
        try:
            spot, exposure = fetch_gex_exposure(ticker)
        except Exception as exc:
            out["ran"] = False
            out["reason"] = str(exc)
            return out

    out["spot"] = float(spot)
    signal_pack = compute_low_gex_signal(exposure, spot=spot)
    out["signal"] = signal_pack
    if not signal_pack.get("available"):
        out["ran"] = True
        out["action"] = "no_signal"
        out["reason"] = signal_pack.get("reason", "No trade signal")
        return out

    rec = signal_pack["recommended"]
    option_type = str(rec["option_type"])
    trade_strike = float(rec["strike"])
    out["recommended"] = rec

    if not execute:
        out["ran"] = True
        out["action"] = "signal_only"
        return out

    rotate = low_gex_reenter_each_bar() if reenter_each_bar is None else reenter_each_bar
    if rotate:
        out["closed_for_rotation"] = _flatten_open_positions(ticker, spot=float(spot))

    if not rotate and len(list_open_trades(ticker)) >= max_open_positions():
        out["ran"] = True
        out["action"] = "skipped"
        out["reason"] = "Max open positions reached"
        return out

    exec_spot = float(spot)
    if _uses_execution_mapping():
        mapped = resolve_execution_spot(signal_spot=float(spot))
        if mapped is None or mapped <= 0:
            out["ran"] = True
            out["action"] = "skipped"
            out["reason"] = "No live execution spot for strike mapping"
            return out
        exec_spot = float(mapped)

    exec_strike = (
        map_execution_strike(trade_strike, signal_spot=float(spot), execution_spot=exec_spot)
        if _uses_execution_mapping()
        else trade_strike
    )

    if not rotate and _has_open_duplicate(ticker, strike=exec_strike, option_type=option_type):
        out["ran"] = True
        out["action"] = "skipped"
        out["reason"] = f"Already open at {exec_strike:.2f} {option_type}"
        return out

    broker = get_broker()
    underlying = webull_underlying() or execution_ticker()
    expire_date = market_today()
    exec_map = (
        execution_summary(signal_strike=trade_strike, signal_spot=float(spot), execution_spot=exec_spot)
        if _uses_execution_mapping()
        else None
    )

    premium_est = estimate_entry_premium(exec_spot, exec_strike)
    qty = resolve_contract_qty(
        confidence=0.65,
        premium=premium_est,
        entry_spot=exec_spot,
        strike=exec_strike,
        account_equity=get_account_equity(),
        size_multiplier=1.0,
    )
    if qty < 1:
        out["ran"] = True
        out["action"] = "skipped"
        out["reason"] = "Risk sizing blocked entry"
        return out

    if live_trading_allowed():
        limit_price = limit_price_for_buy(
            exec_spot,
            exec_strike,
            side="buy",
            underlying=underlying,
            option_type=option_type,
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
            out["ran"] = True
            out["action"] = "order_failed"
            out["reason"] = order.get("stage") or "Order rejected"
            out["order"] = order
            return out
        qty = int(order.get("filled_qty") or qty)
        premium = float(order.get("filled_premium") or order.get("limit_price") or limit_price)
        meta = {
            "paper": False,
            "broker": "webull",
            "strategy": "low_gex",
            "underlying": underlying,
            "expire_date": expire_date,
            "execution_map": exec_map,
            "signal": signal_pack,
            "order": order,
        }
    else:
        premium = premium_est
        meta = {
            "paper": paper_trading_only(),
            "broker": broker.name,
            "strategy": "low_gex",
            "underlying": underlying,
            "expire_date": expire_date,
            "execution_map": exec_map,
            "signal": signal_pack,
        }

    trade_id = open_trade(
        ticker=ticker,
        option_type=option_type,
        strike=exec_strike,
        entry_spot=exec_spot if _uses_execution_mapping() else float(spot),
        entry_premium=premium,
        signal_type="min_gamma_strike",
        signal_strike=trade_strike,
        signal_gamma=float(rec["gamma_bn"]),
        gamma_delta=0.0,
        ai_confidence=0.65,
        ai_reason=str(rec["rationale"]),
        meta=meta,
        qty=float(qty),
    )
    if live_trading_allowed() and meta.get("order", {}).get("client_order_id"):
        patch_trade_meta(trade_id, {"webull_client_order_id": meta["order"]["client_order_id"]})

    record_decision(
        ticker=ticker,
        action="low_gex_entry",
        payload={"signal": signal_pack, "trade_id": trade_id},
        ai_verdict="opened",
        ai_notes=rec["rationale"],
    )

    out.update(
        {
            "ran": True,
            "executed": True,
            "action": "opened",
            "trade_id": trade_id,
            "option_type": option_type,
            "strike": exec_strike,
            "signal_strike": trade_strike,
            "qty": qty,
            "premium": premium,
            "broker": broker.name,
        }
    )
    return out
