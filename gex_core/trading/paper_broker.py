"""Paper option broker — simulates premiums and PnL without live execution."""

from __future__ import annotations

from typing import Any

from gex_core.trading.config import option_leverage


def estimate_entry_premium(spot: float, strike: float) -> float:
    """Rough ATM/near-ATM option premium in index points."""
    if spot <= 0:
        return 1.0
    otm_pct = abs(strike - spot) / spot
    base = spot * 0.0018
    return max(0.5, base * (1.0 + otm_pct * 4.0))


def estimate_option_pnl_pct(
    option_type: str,
    *,
    entry_spot: float,
    current_spot: float,
    strike: float,
) -> float:
    """Paper PnL % on option premium from underlying move."""
    if entry_spot <= 0 or current_spot <= 0:
        return 0.0
    lev = option_leverage()
    if option_type.lower() == "call":
        underlying_ret = (current_spot - entry_spot) / entry_spot
        if strike > entry_spot and current_spot > entry_spot:
            underlying_ret *= 1.0 + min(0.3, (current_spot - entry_spot) / max(strike - entry_spot, 1.0))
    else:
        underlying_ret = (entry_spot - current_spot) / entry_spot
        if strike < entry_spot and current_spot < entry_spot:
            underlying_ret *= 1.0 + min(0.3, (entry_spot - current_spot) / max(entry_spot - strike, 1.0))
    return underlying_ret * lev


def mark_to_market_premium(entry_premium: float, pnl_pct: float) -> float:
    return max(0.01, entry_premium * (1.0 + pnl_pct))


def pnl_usd(entry_premium: float, exit_premium: float, qty: float = 1.0) -> float:
    return (exit_premium - entry_premium) * 100.0 * qty


class PaperBroker:
    name = "paper"

    def buy_option(
        self,
        *,
        underlying: str,
        option_type: str,
        strike: float,
        expire_date: str,
        quantity: int,
        limit_price: float,
        spot: float,
    ) -> dict[str, Any]:
        premium = estimate_entry_premium(spot, strike)
        return {
            "ok": True,
            "broker": self.name,
            "limit_price": premium,
            "filled_premium": premium,
            "client_order_id": None,
        }

    def sell_option(
        self,
        *,
        underlying: str,
        option_type: str,
        strike: float,
        expire_date: str,
        quantity: int,
        limit_price: float,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        return {"ok": True, "broker": self.name, "limit_price": limit_price, "client_order_id": client_order_id}

    def position_pnl_pct(self, trade: dict[str, Any], *, spot: float) -> float | None:
        return estimate_option_pnl_pct(
            trade["option_type"],
            entry_spot=float(trade["entry_spot"]),
            current_spot=spot,
            strike=float(trade["strike"]),
        )
