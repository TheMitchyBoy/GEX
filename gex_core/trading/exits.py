"""Shared exit rules for live trading and backtests."""

from __future__ import annotations

from dataclasses import dataclass

from gex_core.trading.config import (
    dynamic_take_profit_enabled,
    far_otm_distance_pct,
    far_otm_stop_loss_pct,
    stop_loss_pct,
    take_profit_pct,
)


@dataclass
class ExitState:
    peak_pnl_pct: float = 0.0
    partial_taken: bool = False


@dataclass(frozen=True)
class ExitProfile:
    """Per-trade exit thresholds."""

    hold_for_target: bool = True
    partial_take_profit: float | None = None
    trail_trigger: float = 0.10
    trail_floor: float = 0.05
    time_stop_bars: int = 6
    full_take_profit: float = 0.30


def resolve_full_take_profit(profile: ExitProfile, *, expected_move_pct: float | None) -> float:
    """Scale take-profit to expected move when IV data is available."""
    if not dynamic_take_profit_enabled() or not expected_move_pct or expected_move_pct <= 0:
        return profile.full_take_profit
    # Rough option move ≈ 8× underlying expected move for near-ATM 0DTE.
    dynamic = expected_move_pct * 8.0
    return max(0.08, min(profile.full_take_profit, dynamic))


def build_exit_profile(
    *,
    ai_confidence: float,
    gamma_delta: float,
    regime: str | None,
    entry_spot: float,
    strike: float,
    expected_move_pct: float | None = None,
    magnet_strike: float | None = None,
) -> ExitProfile:
    """Exit profile — auto-trader sells at take-profit or gamma strike change only."""
    del ai_confidence, gamma_delta, regime, entry_spot, strike, magnet_strike
    profile = ExitProfile(full_take_profit=take_profit_pct())
    tp = resolve_full_take_profit(profile, expected_move_pct=expected_move_pct)
    if tp != profile.full_take_profit:
        return ExitProfile(full_take_profit=tp)
    return profile


def effective_stop_loss(*, entry_spot: float, strike: float) -> float:
    """Risk sizing helper only — stop-loss exits are disabled in evaluate_exit."""
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


def _gamma_strike_changed(
    entry_positive_gamma_strike: float | None,
    current_positive_gamma_strike: float | None,
) -> bool:
    if entry_positive_gamma_strike is None or current_positive_gamma_strike is None:
        return False
    return abs(entry_positive_gamma_strike - current_positive_gamma_strike) > 0.5


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
    magnet_strike: float | None = None,
    magnet_primary: bool | None = None,
    entry_positive_gamma_strike: float | None = None,
    current_positive_gamma_strike: float | None = None,
) -> tuple[str | None, float]:
    """Return (exit_reason, exit_pnl_pct). Sells only at take-profit or gamma strike change."""
    del bars_held, entry_spot, strike, current_spot, option_type, magnet_strike, magnet_primary
    profile = profile or ExitProfile()
    state.peak_pnl_pct = max(state.peak_pnl_pct, pnl_pct)
    full_tp = profile.full_take_profit

    if pnl_pct >= full_tp:
        return "take_profit", min(pnl_pct, full_tp)

    if _gamma_strike_changed(entry_positive_gamma_strike, current_positive_gamma_strike):
        return "gamma_strike_change", pnl_pct

    return None, pnl_pct
