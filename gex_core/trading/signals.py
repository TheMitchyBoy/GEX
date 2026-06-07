"""Gamma-based entry signals for the auto-trader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from gex_core.trading.config import (
    max_strike_distance_pct,
    min_fastest_gamma_delta,
    min_gamma_delta,
    multi_strike_count,
    prefer_signal_type,
)


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
    """Pick nearest ATM/ slightly ITM positive-gamma strike within max distance."""
    if spot <= 0:
        return None
    positive = cur[cur > 0]
    if positive.empty:
        return None

    max_dist = max_strike_distance_pct()
    candidates: list[tuple[float, float, bool]] = []
    for strike_val, gamma_val in positive.items():
        strike_f = float(strike_val)
        if float(gamma_val) <= 0:
            continue
        dist = abs(strike_f - spot) / spot
        if dist > max_dist:
            continue
        if option_type == "call" and strike_f >= spot * 0.998:
            itm = strike_f <= spot
            candidates.append((dist, strike_f, itm))
        elif option_type == "put" and strike_f <= spot * 1.002:
            itm = strike_f >= spot
            candidates.append((dist, strike_f, itm))

    if candidates:
        candidates.sort(key=lambda row: (row[0], 0 if abs(row[1] - spot) < spot * 0.0005 else 1, 0 if row[2] else 1))
        return candidates[0][1]

    dist_magnet = abs(magnet_strike - spot) / spot
    if dist_magnet <= max_dist:
        return float(magnet_strike)
    return None


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


def _candidate_key(rec: dict[str, Any]) -> tuple[float, str]:
    return (round(float(rec["strike"]), 2), str(rec["option_type"]).lower())


def _rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred = prefer_signal_type()

    def sort_key(rec: dict[str, Any]) -> tuple:
        sig = str(rec.get("signal_type", "")).lower()
        pref_rank = 0 if preferred and sig == preferred else 1
        delta = float(rec.get("gamma_delta", 0.0))
        score = float(rec.get("score", 0.0))
        return (pref_rank, -delta, -score)

    return sorted(candidates, key=sort_key)


def compute_entry_candidates(
    exposure: pd.Series | None,
    previous: pd.Series | None,
    *,
    spot: float | None,
) -> dict[str, Any]:
    """Return up to N near-spot positive-gamma entry candidates per bar."""
    cur = _clean(exposure)
    prev = _clean(previous)
    spot_val = float(spot or 0.0)
    limit = multi_strike_count()

    if cur.empty:
        return {"available": False, "reason": "No gamma exposure data"}

    positive = cur[cur > 0]
    max_pos_strike = float(positive.idxmax()) if not positive.empty else float(cur.idxmax())
    max_pos_gamma = float(positive.max()) if not positive.empty else float(cur.max())
    max_pos_delta = 0.0

    delta = cur.subtract(prev.reindex(cur.index), fill_value=0.0) if not prev.empty else cur * 0.0
    fastest_strike = float(delta.idxmax()) if not delta.empty else max_pos_strike
    fastest_delta = float(delta.max()) if not delta.empty else 0.0
    fastest_gamma = float(cur.get(fastest_strike, 0.0))
    if not positive.empty:
        max_pos_delta = float(delta.get(max_pos_strike, 0.0))

    max_option_type = _option_type_for_strike(max_pos_strike, spot_val)
    max_trade_strike = _refine_trade_strike(cur, max_pos_strike, spot_val, max_option_type)

    max_pos_signal = GammaSignal(
        signal_type="max_positive_gamma",
        strike=max_trade_strike or max_pos_strike,
        gamma_bn=max_pos_gamma,
        gamma_delta=max_pos_delta,
        score=max_pos_gamma,
        option_type=max_option_type,
        rationale=f"Largest positive gamma magnet {max_pos_strike:.0f}",
    )

    fast_option_type = _option_type_for_strike(fastest_strike, spot_val)
    fast_trade_strike = _refine_trade_strike(cur, fastest_strike, spot_val, fast_option_type)
    fastest_signal = GammaSignal(
        signal_type="fastest_gamma_increase",
        strike=fast_trade_strike or fastest_strike,
        gamma_bn=fastest_gamma,
        gamma_delta=fastest_delta,
        score=fastest_delta + max(fastest_gamma, 0.0) * 0.25,
        option_type=fast_option_type,
        rationale=f"Fastest gamma increase at {fastest_strike:.0f} (Δ{fastest_delta:+.3f} Bn)",
    )

    seen: set[tuple[float, str]] = set()
    candidates: list[dict[str, Any]] = []
    min_delta = min_gamma_delta()

    for magnet_strike, gamma_bn in positive.sort_values(ascending=False).items():
        if len(candidates) >= limit:
            break
        magnet_f = float(magnet_strike)
        magnet_delta = float(delta.get(magnet_strike, 0.0))
        if magnet_delta < 0:
            continue
        option_type = _option_type_for_strike(magnet_f, spot_val)
        trade_strike = _refine_trade_strike(cur, magnet_f, spot_val, option_type)
        if trade_strike is None:
            continue
        sig = GammaSignal(
            signal_type="max_positive_gamma",
            strike=trade_strike,
            gamma_bn=float(gamma_bn),
            gamma_delta=magnet_delta,
            score=float(gamma_bn),
            option_type=option_type,
            rationale=f"Positive gamma magnet {magnet_f:.0f}, trading {trade_strike:.0f}",
        )
        rec = _signal_dict(sig)
        if (
            fast_trade_strike is not None
            and fastest_delta >= min_delta
            and _candidate_key(rec) == _candidate_key(_signal_dict(fastest_signal))
        ):
            rec = {**_signal_dict(fastest_signal), "strike": rec["strike"]}
        key = _candidate_key(rec)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(rec)

    min_fast = min_fastest_gamma_delta()
    fast_rec = _signal_dict(fastest_signal)
    if fast_trade_strike is not None and fastest_delta >= min_delta:
        fast_dist = abs(fastest_strike - spot_val) / spot_val if spot_val > 0 else 1.0
        if fast_dist <= max_strike_distance_pct() and _candidate_key(fast_rec) not in seen:
            seen.add(_candidate_key(fast_rec))
            candidates.append(fast_rec)

    if not candidates and max_pos_delta < 0:
        if fastest_delta >= min_fast and fast_trade_strike is not None:
            fast_dist = abs(fastest_strike - spot_val) / spot_val if spot_val > 0 else 1.0
            if fast_dist <= max_strike_distance_pct():
                candidates.append(_signal_dict(fastest_signal))
        if not candidates:
            return {
                "available": False,
                "reason": (
                    f"Largest positive gamma at {max_pos_strike:.0f} declined "
                    f"(Δ{max_pos_delta:+.3f} Bn) and no near-spot magnets are flat or rising"
                ),
                "skip_reason": "gamma_declined",
                "spot": spot_val,
                "max_positive_gamma": _signal_dict(max_pos_signal),
                "fastest_gamma_increase": _signal_dict(fastest_signal),
                "max_pos_gamma_delta": max_pos_delta,
            }

    if not candidates:
        return {
            "available": False,
            "reason": f"No positive-gamma strike within {max_strike_distance_pct():.1%} of spot",
            "skip_reason": "strike_too_far",
            "spot": spot_val,
            "max_positive_gamma": _signal_dict(max_pos_signal),
            "fastest_gamma_increase": _signal_dict(fastest_signal),
        }

    candidates = _rank_candidates(candidates)[:limit]
    selection_reason = "max_positive_gamma"
    recommended_dict = candidates[0]
    if max_pos_delta < 0 and recommended_dict.get("signal_type") == "fastest_gamma_increase":
        selection_reason = "max_positive_gamma_declined"

    return {
        "available": True,
        "spot": spot_val,
        "selection_reason": selection_reason,
        "candidates": candidates,
        "recommended": recommended_dict,
        "max_positive_gamma": _signal_dict(max_pos_signal),
        "fastest_gamma_increase": _signal_dict(fastest_signal),
        "max_pos_gamma_delta": max_pos_delta,
        "gamma_delta_by_strike": {float(k): float(v) for k, v in delta.nlargest(8).items()},
    }


def compute_gamma_signals(
    exposure: pd.Series | None,
    previous: pd.Series | None,
    *,
    spot: float | None,
) -> dict[str, Any]:
    """Primary gamma signal bundle (first entry candidate)."""
    return compute_entry_candidates(exposure, previous, spot=spot)
