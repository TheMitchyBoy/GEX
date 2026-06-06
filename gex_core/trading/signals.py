"""Gamma-based entry signals for the auto-trader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from gex_core.trading.config import max_strike_distance_pct, min_fastest_gamma_delta


@dataclass(frozen=True)
class GammaSignal:
    signal_type: str
    strike: float
    gamma_bn: float
    gamma_delta: float
    score: float
    option_type: str
    rationale: str


def _clean(series: pd.Series | None) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    s = pd.Series(pd.to_numeric(series, errors="coerce"), index=pd.to_numeric(series.index, errors="coerce"))
    s = s.dropna()
    if s.index.duplicated().any():
        s = s.groupby(level=0).sum()
    return s.sort_index()


def _option_type_for_strike(strike: float, spot: float) -> str:
    return "call" if strike >= spot else "put"


def _refine_trade_strike(cur: pd.Series, magnet_strike: float, spot: float, option_type: str) -> float | None:
    """Pick the nearest positive-gamma strike toward spot within max distance."""
    if spot <= 0:
        return None
    positive = cur[cur > 0]
    if positive.empty:
        return None

    max_dist = max_strike_distance_pct()
    candidates: list[tuple[float, float]] = []
    for strike_val, gamma_val in positive.items():
        strike_f = float(strike_val)
        if float(gamma_val) <= 0:
            continue
        dist = abs(strike_f - spot) / spot
        if dist > max_dist:
            continue
        if option_type == "call" and strike_f >= spot:
            candidates.append((dist, strike_f))
        elif option_type == "put" and strike_f <= spot:
            candidates.append((dist, strike_f))

    if candidates:
        candidates.sort(key=lambda row: row[0])
        return candidates[0][1]

    dist_magnet = abs(magnet_strike - spot) / spot
    if dist_magnet <= max_dist:
        return float(magnet_strike)
    return None


def compute_gamma_signals(
    exposure: pd.Series | None,
    previous: pd.Series | None,
    *,
    spot: float | None,
) -> dict[str, Any]:
    """Rank strikes by max positive gamma and fastest gamma increase."""
    cur = _clean(exposure)
    prev = _clean(previous)
    spot_val = float(spot or 0.0)

    if cur.empty:
        return {"available": False, "reason": "No gamma exposure data"}

    positive = cur[cur > 0]
    max_pos_strike = float(positive.idxmax()) if not positive.empty else float(cur.idxmax())
    max_pos_gamma = float(positive.max()) if not positive.empty else float(cur.max())

    delta = cur.subtract(prev.reindex(cur.index), fill_value=0.0) if not prev.empty else cur * 0.0
    fastest_strike = float(delta.idxmax()) if not delta.empty else max_pos_strike
    fastest_delta = float(delta.max()) if not delta.empty else 0.0
    fastest_gamma = float(cur.get(fastest_strike, 0.0))
    max_pos_delta = float(delta.get(max_pos_strike, 0.0))

    max_option_type = _option_type_for_strike(max_pos_strike, spot_val)
    max_trade_strike = _refine_trade_strike(cur, max_pos_strike, spot_val, max_option_type)
    if max_trade_strike is None:
        return {
            "available": False,
            "reason": f"No positive-gamma strike within {max_strike_distance_pct():.1%} of spot",
            "skip_reason": "strike_too_far",
            "spot": spot_val,
        }

    max_pos_signal = GammaSignal(
        signal_type="max_positive_gamma",
        strike=max_trade_strike,
        gamma_bn=max_pos_gamma,
        gamma_delta=max_pos_delta,
        score=max_pos_gamma,
        option_type=max_option_type,
        rationale=f"Largest positive gamma magnet {max_pos_strike:.0f}, trading {max_trade_strike:.0f}",
    )

    fast_option_type = _option_type_for_strike(fastest_strike, spot_val)
    fast_trade_strike = _refine_trade_strike(cur, fastest_strike, spot_val, fast_option_type)
    accel_score = fastest_delta + max(fastest_gamma, 0.0) * 0.25
    fastest_signal = GammaSignal(
        signal_type="fastest_gamma_increase",
        strike=fast_trade_strike or fastest_strike,
        gamma_bn=fastest_gamma,
        gamma_delta=fastest_delta,
        score=accel_score,
        option_type=fast_option_type,
        rationale=f"Fastest 10m gamma increase at {fastest_strike:.0f} (Δ{fastest_delta:+.3f} Bn)",
    )

    selection_reason = "max_positive_gamma"
    if max_pos_delta >= 0:
        recommended = max_pos_signal
    elif fastest_delta >= min_fastest_gamma_delta():
        max_dist = abs(max_pos_strike - spot_val) / spot_val if spot_val > 0 else 1.0
        fast_dist = abs(fastest_strike - spot_val) / spot_val if spot_val > 0 else 1.0
        if fast_trade_strike is None or fast_dist > max_strike_distance_pct():
            return {
                "available": False,
                "reason": "Max gamma declined and fastest increase strike is too far from spot",
                "skip_reason": "strike_too_far",
                "spot": spot_val,
                "max_positive_gamma": _signal_dict(max_pos_signal),
                "fastest_gamma_increase": _signal_dict(fastest_signal),
            }
        if fast_dist > max_dist:
            return {
                "available": False,
                "reason": "Max gamma declined but fastest increase is not closer to spot",
                "skip_reason": "gamma_declined",
                "spot": spot_val,
                "max_positive_gamma": _signal_dict(max_pos_signal),
                "fastest_gamma_increase": _signal_dict(fastest_signal),
            }
        recommended = fastest_signal
        selection_reason = "max_positive_gamma_declined"
    else:
        return {
            "available": False,
            "reason": (
                f"Largest positive gamma at {max_pos_strike:.0f} declined "
                f"(Δ{max_pos_delta:+.3f} Bn) and no strike shows sufficient rising gamma"
            ),
            "skip_reason": "gamma_declined",
            "spot": spot_val,
            "max_positive_gamma": _signal_dict(max_pos_signal),
            "fastest_gamma_increase": _signal_dict(fastest_signal),
            "max_pos_gamma_delta": max_pos_delta,
        }

    return {
        "available": True,
        "spot": spot_val,
        "selection_reason": selection_reason,
        "max_positive_gamma": _signal_dict(max_pos_signal),
        "fastest_gamma_increase": _signal_dict(fastest_signal),
        "recommended": _signal_dict(recommended),
        "max_pos_gamma_delta": max_pos_delta,
        "gamma_delta_by_strike": {float(k): float(v) for k, v in delta.nlargest(8).items()},
    }


def _signal_dict(sig: GammaSignal) -> dict[str, Any]:
    return {
        "signal_type": sig.signal_type,
        "strike": sig.strike,
        "gamma_bn": sig.gamma_bn,
        "gamma_delta": sig.gamma_delta,
        "score": sig.score,
        "option_type": sig.option_type,
        "rationale": sig.rationale,
    }
