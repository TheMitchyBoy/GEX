"""Position sizing — confidence base with optional risk-per-trade cap."""

from __future__ import annotations

from gex_core.trading.config import (
    account_equity_usd,
    high_confidence_contracts,
    risk_per_trade_pct,
    use_risk_based_sizing,
    webull_contracts,
)
from gex_core.trading.exits import contracts_for_confidence, effective_stop_loss


def resolve_contract_qty(
    *,
    confidence: float,
    premium: float,
    entry_spot: float,
    strike: float,
    account_equity: float | None = None,
    size_multiplier: float = 1.0,
) -> int:
    """Contracts from confidence, capped by risk budget and buying power."""
    desired = int(contracts_for_confidence(confidence))
    desired = max(1, min(desired, high_confidence_contracts()))

    if size_multiplier < 1.0:
        desired = max(1, int(desired * size_multiplier))

    if not use_risk_based_sizing() or premium <= 0:
        return desired

    equity = float(account_equity if account_equity is not None else account_equity_usd())
    risk_budget = equity * risk_per_trade_pct()
    stop = effective_stop_loss(entry_spot=entry_spot, strike=strike)
    unit_risk = premium * 100.0 * stop
    if unit_risk <= 0:
        return desired

    risk_qty = int(risk_budget // unit_risk)
    if risk_qty < 1:
        return 0
    return max(1, min(desired, risk_qty))


def affordable_qty(premium: float, cash: float, desired: float) -> int:
    unit_cost = premium * 100.0
    if unit_cost <= 0 or cash <= 0:
        return 0
    return max(0, min(int(desired), int(cash // unit_cost)))
