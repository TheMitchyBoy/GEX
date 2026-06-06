"""Shared exit rules for live trading and backtests."""

from __future__ import annotations

from dataclasses import dataclass

from gex_core.trading.config import (
    far_otm_distance_pct,
    far_otm_stop_loss_pct,
    partial_take_profit_pct,
    stop_loss_pct,
    take_profit_pct,
    time_stop_bars,
    time_stop_min_pnl_pct,
    trailing_stop_floor_pct,
    trailing_stop_trigger_pct,
)


@dataclass
class ExitState:
    peak_pnl_pct: float = 0.0
    partial_taken: bool = False


def effective_stop_loss(*, entry_spot: float, strike: float) -> float:
    if entry_spot <= 0:
        return stop_loss_pct()
    dist = abs(strike - entry_spot) / entry_spot
    if dist > far_otm_distance_pct():
        return far_otm_stop_loss_pct()
    return stop_loss_pct()


def evaluate_exit(
    pnl_pct: float,
    *,
    state: ExitState,
    bars_held: int,
    entry_spot: float,
    strike: float,
) -> tuple[str | None, float]:
    """Return (exit_reason, exit_pnl_pct)."""
    state.peak_pnl_pct = max(state.peak_pnl_pct, pnl_pct)
    stop = effective_stop_loss(entry_spot=entry_spot, strike=strike)

    if pnl_pct <= -stop:
        return "stop_loss", max(pnl_pct, -stop)

    if not state.partial_taken and pnl_pct >= partial_take_profit_pct():
        state.partial_taken = True
        return "take_profit_partial", min(pnl_pct, partial_take_profit_pct())

    if pnl_pct >= take_profit_pct():
        return "take_profit", min(pnl_pct, take_profit_pct())

    if state.peak_pnl_pct >= trailing_stop_trigger_pct() and pnl_pct <= trailing_stop_floor_pct():
        return "trailing_stop", max(pnl_pct, trailing_stop_floor_pct())

    if bars_held >= time_stop_bars() and pnl_pct < time_stop_min_pnl_pct():
        return "time_stop", pnl_pct

    return None, pnl_pct
