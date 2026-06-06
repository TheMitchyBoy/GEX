"""Shared exit rules for live trading and backtests."""

from __future__ import annotations

from dataclasses import dataclass

from gex_core.trading.config import (
    far_otm_distance_pct,
    far_otm_stop_loss_pct,
    magnet_proximity_pct,
    partial_take_profit_pct,
    stop_loss_pct,
    strong_entry_confidence,
    strong_gamma_delta,
    take_profit_pct,
    time_stop_bars,
    time_stop_min_magnet_progress,
    time_stop_min_pnl_pct,
    trailing_stop_floor_pct,
    trailing_stop_trigger_pct,
)


@dataclass
class ExitState:
    peak_pnl_pct: float = 0.0
    partial_taken: bool = False


@dataclass(frozen=True)
class ExitProfile:
    """Per-trade exit thresholds — strong setups run to full target."""

    hold_for_target: bool = False
    partial_take_profit: float | None = None
    trail_trigger: float = 0.10
    trail_floor: float = 0.05
    time_stop_bars: int = 6
    full_take_profit: float = 0.35


def build_exit_profile(
    *,
    ai_confidence: float,
    gamma_delta: float,
    regime: str | None,
    entry_spot: float,
    strike: float,
) -> ExitProfile:
    """Strong gamma + confidence setups skip early partials and use wider trails."""
    near_magnet = False
    if entry_spot > 0:
        near_magnet = abs(strike - entry_spot) / entry_spot <= magnet_proximity_pct()

    strong = ai_confidence >= strong_entry_confidence() and gamma_delta >= strong_gamma_delta()
    short_gamma = "SHORT" in (regime or "").upper()

    if strong or near_magnet:
        return ExitProfile(
            hold_for_target=True,
            partial_take_profit=None,
            trail_trigger=0.15,
            trail_floor=0.08,
            time_stop_bars=12,
            full_take_profit=take_profit_pct(),
        )

    if short_gamma:
        return ExitProfile(
            hold_for_target=False,
            partial_take_profit=partial_take_profit_pct(),
            trail_trigger=0.12,
            trail_floor=0.06,
            time_stop_bars=8,
            full_take_profit=take_profit_pct(),
        )

    return ExitProfile(
        hold_for_target=False,
        partial_take_profit=partial_take_profit_pct(),
        trail_trigger=trailing_stop_trigger_pct(),
        trail_floor=trailing_stop_floor_pct(),
        time_stop_bars=time_stop_bars(),
        full_take_profit=take_profit_pct(),
    )


def effective_stop_loss(*, entry_spot: float, strike: float) -> float:
    if entry_spot <= 0:
        return stop_loss_pct()
    dist = abs(strike - entry_spot) / entry_spot
    if dist > far_otm_distance_pct():
        return far_otm_stop_loss_pct()
    return stop_loss_pct()


def spot_progress_toward_strike(
    *,
    entry_spot: float,
    current_spot: float,
    strike: float,
    option_type: str,
) -> float:
    """Fraction of entry→strike distance closed (0 = none, 1 = at strike)."""
    if entry_spot <= 0 or strike <= 0:
        return 0.0
    opt = option_type.lower()
    if opt == "call":
        if strike <= entry_spot:
            return 1.0 if current_spot >= strike else 0.0
        gap0 = strike - entry_spot
        gap1 = strike - current_spot
    else:
        if strike >= entry_spot:
            return 1.0 if current_spot <= strike else 0.0
        gap0 = entry_spot - strike
        gap1 = current_spot - strike
    if gap0 <= 0:
        return 0.0
    return max(0.0, min(1.0, (gap0 - gap1) / gap0))


def contracts_for_confidence(ai_confidence: float) -> float:
    from gex_core.trading.config import high_confidence_contracts, strong_entry_confidence, webull_contracts

    base = float(webull_contracts())
    if ai_confidence >= strong_entry_confidence():
        return float(max(base, high_confidence_contracts()))
    return base


def evaluate_exit(
    pnl_pct: float,
    *,
    state: ExitState,
    bars_held: int,
    entry_spot: float,
    strike: float,
    current_spot: float,
    option_type: str,
    profile: ExitProfile | None = None,
) -> tuple[str | None, float]:
    """Return (exit_reason, exit_pnl_pct)."""
    profile = profile or ExitProfile()
    state.peak_pnl_pct = max(state.peak_pnl_pct, pnl_pct)
    stop = effective_stop_loss(entry_spot=entry_spot, strike=strike)

    if pnl_pct <= -stop:
        return "stop_loss", max(pnl_pct, -stop)

    if pnl_pct >= profile.full_take_profit:
        return "take_profit", min(pnl_pct, profile.full_take_profit)

    if (
        not profile.hold_for_target
        and not state.partial_taken
        and profile.partial_take_profit is not None
        and pnl_pct >= profile.partial_take_profit
    ):
        state.partial_taken = True
        return "take_profit_partial", min(pnl_pct, profile.partial_take_profit)

    if state.peak_pnl_pct >= profile.trail_trigger and pnl_pct <= profile.trail_floor:
        return "trailing_stop", max(pnl_pct, profile.trail_floor)

    if bars_held >= profile.time_stop_bars and pnl_pct < time_stop_min_pnl_pct():
        progress = spot_progress_toward_strike(
            entry_spot=entry_spot,
            current_spot=current_spot,
            strike=strike,
            option_type=option_type,
        )
        if progress < time_stop_min_magnet_progress():
            return "time_stop", pnl_pct

    return None, pnl_pct
