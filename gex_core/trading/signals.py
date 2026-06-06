"""Gamma-based entry signals for the auto-trader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


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


def compute_gamma_signals(
    exposure: pd.Series | None,
    previous: pd.Series | None,
    *,
    spot: float | None,
) -> dict[str, Any]:
    """Rank strikes by max positive gamma and fastest gamma increase.

    Entry rule:
    - Default to the largest positive gamma strike when gamma there is flat or rising.
    - If gamma at that strike declined since the prior snapshot, switch to the fastest
      gamma increase strike when it is still rising.
    - Skip entry when the max-gamma strike declined and no strike shows rising gamma.
    """
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

    max_pos_signal = GammaSignal(
        signal_type="max_positive_gamma",
        strike=max_pos_strike,
        gamma_bn=max_pos_gamma,
        gamma_delta=max_pos_delta,
        score=max_pos_gamma,
        option_type="call" if max_pos_strike >= spot_val else "put",
        rationale=f"Largest positive gamma at {max_pos_strike:.0f} ({max_pos_gamma:+.3f} Bn)",
    )

    accel_score = fastest_delta + max(fastest_gamma, 0.0) * 0.25
    fastest_signal = GammaSignal(
        signal_type="fastest_gamma_increase",
        strike=fastest_strike,
        gamma_bn=fastest_gamma,
        gamma_delta=fastest_delta,
        score=accel_score,
        option_type="call" if fastest_strike >= spot_val else "put",
        rationale=f"Fastest 10m gamma increase at {fastest_strike:.0f} (Δ{fastest_delta:+.3f} Bn)",
    )

    selection_reason = "max_positive_gamma"
    if max_pos_delta >= 0:
        recommended = max_pos_signal
    elif fastest_delta > 0:
        recommended = fastest_signal
        selection_reason = "max_positive_gamma_declined"
    else:
        return {
            "available": False,
            "reason": (
                f"Largest positive gamma at {max_pos_strike:.0f} declined "
                f"(Δ{max_pos_delta:+.3f} Bn) and no strike shows rising gamma"
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
