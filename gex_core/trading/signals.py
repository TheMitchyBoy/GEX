"""Gamma-based entry signals for the auto-trader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from gex_core.features import select_atm_strike_series
from gex_core.trading.config import (
    clear_all_filters,
    magnet_anchored_strikes,
    max_gamma_only,
    max_strike_distance_pct,
    min_fastest_gamma_delta,
    min_gamma_delta,
    multi_strike_count,
    prefer_signal_type,
    trade_negative_gamma_magnets,
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
    magnet_strike: float | None = None


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


def _magnet_anchored_trade_strike(
    cur: pd.Series,
    magnet_strike: float,
    spot: float,
    option_type: str,
) -> float | None:
    """Trade at the gamma magnet strike when within max distance of spot."""
    _ = (cur, option_type)
    if spot <= 0:
        return None
    max_dist = max_strike_distance_pct()
    dist = abs(magnet_strike - spot) / spot
    if dist > max_dist:
        return None
    return float(magnet_strike)


def _resolve_trade_strike(
    cur: pd.Series,
    magnet_strike: float,
    spot: float,
    option_type: str,
) -> float | None:
    if magnet_anchored_strikes():
        return _magnet_anchored_trade_strike(cur, magnet_strike, spot, option_type)
    return _refine_trade_strike(cur, magnet_strike, spot, option_type)


def _extreme_gamma_magnet(search: pd.Series) -> tuple[float, float, str] | None:
    """Pick the trade magnet: highest +γ, or lowest −γ when enabled and dominant."""
    if search.empty:
        return None

    positive = search[search > 0]
    negative = search[search < 0]
    max_pos_strike = float(positive.idxmax()) if not positive.empty else None
    max_pos_gamma = float(positive.max()) if not positive.empty else 0.0
    min_neg_strike = float(negative.idxmin()) if not negative.empty else None
    min_neg_gamma = float(negative.min()) if not negative.empty else 0.0

    if not trade_negative_gamma_magnets():
        if max_pos_strike is None:
            return None
        return max_pos_strike, max_pos_gamma, "max_positive_gamma"

    pos_mag = abs(max_pos_gamma) if max_pos_strike is not None else 0.0
    neg_mag = abs(min_neg_gamma) if min_neg_strike is not None else 0.0

    if neg_mag > pos_mag and min_neg_strike is not None:
        return min_neg_strike, min_neg_gamma, "min_negative_gamma"
    if max_pos_strike is not None:
        return max_pos_strike, max_pos_gamma, "max_positive_gamma"
    if min_neg_strike is not None:
        return min_neg_strike, min_neg_gamma, "min_negative_gamma"

    strike = float(search.idxmax())
    gamma = float(search.loc[strike])
    sig_type = "max_positive_gamma" if gamma >= 0 else "min_negative_gamma"
    return strike, gamma, sig_type


def _resolve_extreme_trade_strike(
    cur: pd.Series,
    magnet_strike: float,
    spot: float,
    option_type: str,
    gamma_bn: float,
) -> float | None:
    """Trade strike for dominant +γ or −γ magnets (negative walls trade at the magnet)."""
    if gamma_bn < 0:
        if spot <= 0:
            return None
        dist = abs(magnet_strike - spot) / spot
        if dist <= max_strike_distance_pct():
            return float(magnet_strike)
        return None
    return _resolve_trade_strike(cur, magnet_strike, spot, option_type)


def _signal_dict(sig: GammaSignal) -> dict[str, Any]:
    return {
        "signal_type": sig.signal_type,
        "strike": sig.strike,
        "gamma_bn": sig.gamma_bn,
        "gamma_delta": sig.gamma_delta,
        "score": sig.score,
        "option_type": sig.option_type,
        "rationale": sig.rationale,
        "magnet_strike": sig.magnet_strike if sig.magnet_strike is not None else sig.strike,
    }


def _candidate_key(rec: dict[str, Any]) -> tuple[float, str]:
    return (round(float(rec["strike"]), 2), str(rec["option_type"]).lower())


def signal_performance_weights(ticker: str | None = None) -> dict[str, float]:
    """Map signal_type → score multiplier from paper-trade PnL history."""
    try:
        from gex_core.trading.journal import get_performance_summary

        perf = get_performance_summary(ticker)
        weights: dict[str, float] = {}
        for sig, stats in (perf.get("by_signal") or {}).items():
            count = int(stats.get("count") or 0)
            if count < 3:
                weights[sig] = 1.0
                continue
            win_rate = float(stats.get("win_rate") or 0.5)
            avg_pnl = float(stats.get("avg_pnl_pct") or 0.0)
            weight = 0.85 + 0.3 * win_rate + max(-0.15, min(0.15, avg_pnl * 2.0))
            weights[sig] = max(0.5, min(1.5, weight))
        return weights
    except Exception:
        return {}


def _apply_performance_weight(rec: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    if not weights:
        return rec
    sig = str(rec.get("signal_type") or "")
    multiplier = weights.get(sig, 1.0)
    if multiplier == 1.0:
        return rec
    out = dict(rec)
    out["score"] = float(rec.get("score", 0.0)) * multiplier
    out["performance_weight"] = multiplier
    return out


def _rank_candidates(candidates: list[dict[str, Any]], *, weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
    preferred = prefer_signal_type()
    weights = weights or {}

    def sort_key(rec: dict[str, Any]) -> tuple:
        sig = str(rec.get("signal_type", "")).lower()
        pref_rank = 0 if preferred and sig == preferred else 1
        delta = float(rec.get("gamma_delta", 0.0))
        score = float(rec.get("score", 0.0))
        return (pref_rank, -delta, -score)

    weighted = [_apply_performance_weight(rec, weights) for rec in candidates]
    return sorted(weighted, key=sort_key)


def _unavailable(
    *,
    reason: str,
    skip_reason: str,
    spot_val: float,
    max_pos_signal: GammaSignal,
    fastest_signal: GammaSignal,
    max_pos_delta: float,
    min_neg_signal: GammaSignal | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "reason": reason,
        "skip_reason": skip_reason,
        "spot": spot_val,
        "max_positive_gamma": _signal_dict(max_pos_signal),
        "fastest_gamma_increase": _signal_dict(fastest_signal),
        "max_pos_gamma_delta": max_pos_delta,
    }
    if min_neg_signal is not None:
        out["min_negative_gamma"] = _signal_dict(min_neg_signal)
    return out


def _compute_max_gamma_only(
    *,
    cur: pd.Series,
    search: pd.Series,
    spot_val: float,
    delta: pd.Series,
    max_pos_signal: GammaSignal,
    min_neg_signal: GammaSignal | None,
    fastest_signal: GammaSignal,
    max_pos_delta: float,
    perf_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Single dominant |gamma| candidate — highest +γ or lowest −γ near spot."""
    perf_weights = perf_weights or {}
    extreme = _extreme_gamma_magnet(search)
    if extreme is None:
        return _unavailable(
            reason="No positive gamma magnet near spot",
            skip_reason="no_positive_gamma",
            spot_val=spot_val,
            max_pos_signal=max_pos_signal,
            fastest_signal=fastest_signal,
            max_pos_delta=max_pos_delta,
            min_neg_signal=min_neg_signal,
        )

    magnet_strike, magnet_gamma, magnet_sig_type = extreme
    magnet_delta = float(delta.get(magnet_strike, 0.0))
    magnet_label = "Highest" if magnet_sig_type == "max_positive_gamma" else "Lowest"

    if magnet_delta < 0 and not clear_all_filters():
        return _unavailable(
            reason=(
                f"{magnet_label} gamma at {magnet_strike:.0f} declined "
                f"(Δ{magnet_delta:+.3f} Bn)"
            ),
            skip_reason="gamma_declined",
            spot_val=spot_val,
            max_pos_signal=max_pos_signal,
            fastest_signal=fastest_signal,
            max_pos_delta=max_pos_delta,
            min_neg_signal=min_neg_signal,
        )

    master_direction = _option_type_for_strike(magnet_strike, spot_val)
    trade_strike = _resolve_extreme_trade_strike(
        cur, magnet_strike, spot_val, master_direction, magnet_gamma
    )
    if trade_strike is None:
        return _unavailable(
            reason=f"No tradeable strike within {max_strike_distance_pct():.1%} of spot",
            skip_reason="strike_too_far",
            spot_val=spot_val,
            max_pos_signal=max_pos_signal,
            fastest_signal=fastest_signal,
            max_pos_delta=max_pos_delta,
            min_neg_signal=min_neg_signal,
        )

    base_score = abs(magnet_gamma) * perf_weights.get(magnet_sig_type, 1.0)
    sig = GammaSignal(
        signal_type=magnet_sig_type,
        strike=trade_strike,
        gamma_bn=magnet_gamma,
        gamma_delta=magnet_delta,
        score=base_score,
        option_type=master_direction,
        rationale=(
            f"{magnet_label} gamma {magnet_gamma:+.3f} Bn at {magnet_strike:.0f} "
            f"→ {master_direction}"
        ),
        magnet_strike=magnet_strike,
    )
    rec = _signal_dict(sig)
    out: dict[str, Any] = {
        "available": True,
        "spot": spot_val,
        "selection_reason": magnet_sig_type,
        "candidates": [rec],
        "recommended": rec,
        "max_positive_gamma": _signal_dict(max_pos_signal),
        "fastest_gamma_increase": _signal_dict(fastest_signal),
        "max_pos_gamma_delta": max_pos_delta,
        "master_direction": master_direction,
        "gamma_delta_by_strike": {},
    }
    if min_neg_signal is not None:
        out["min_negative_gamma"] = _signal_dict(min_neg_signal)
    return out


def _compute_legacy_candidates(
    *,
    cur: pd.Series,
    positive: pd.Series,
    spot_val: float,
    max_pos_strike: float,
    max_pos_delta: float,
    max_pos_signal: GammaSignal,
    fastest_signal: GammaSignal,
    fastest_strike: float,
    fastest_delta: float,
    fast_trade_strike: float | None,
    delta: pd.Series,
    perf_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    limit = multi_strike_count()
    perf_weights = perf_weights or {}
    master_direction = _option_type_for_strike(max_pos_strike, spot_val)
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
        option_type = master_direction
        trade_strike = _resolve_trade_strike(cur, magnet_f, spot_val, option_type)
        if trade_strike is None:
            continue
        sig = GammaSignal(
            signal_type="max_positive_gamma",
            strike=trade_strike,
            gamma_bn=float(gamma_bn),
            gamma_delta=magnet_delta,
            score=float(gamma_bn) * perf_weights.get("max_positive_gamma", 1.0),
            option_type=option_type,
            rationale=f"Positive gamma magnet {magnet_f:.0f}, trading {trade_strike:.0f}",
            magnet_strike=magnet_f,
        )
        rec = _signal_dict(sig)
        if (
            fast_trade_strike is not None
            and fastest_delta >= min_delta
            and _candidate_key(rec) == _candidate_key(_signal_dict(fastest_signal))
        ):
            rec = {**_signal_dict(fastest_signal), "strike": rec["strike"], "magnet_strike": magnet_f}
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
            if _option_type_for_strike(fastest_strike, spot_val) == master_direction:
                seen.add(_candidate_key(fast_rec))
                candidates.append(fast_rec)

    if not candidates and max_pos_delta < 0:
        if fastest_delta >= min_fast and fast_trade_strike is not None:
            fast_dist = abs(fastest_strike - spot_val) / spot_val if spot_val > 0 else 1.0
            if (
                fast_dist <= max_strike_distance_pct()
                and _option_type_for_strike(fastest_strike, spot_val) == master_direction
            ):
                candidates.append(_signal_dict(fastest_signal))
        if not candidates:
            return _unavailable(
                reason=(
                    f"Largest positive gamma at {max_pos_strike:.0f} declined "
                    f"(Δ{max_pos_delta:+.3f} Bn) and no near-spot magnets are flat or rising"
                ),
                skip_reason="gamma_declined",
                spot_val=spot_val,
                max_pos_signal=max_pos_signal,
                fastest_signal=fastest_signal,
                max_pos_delta=max_pos_delta,
            )

    if not candidates:
        return _unavailable(
            reason=f"No positive-gamma strike within {max_strike_distance_pct():.1%} of spot",
            skip_reason="strike_too_far",
            spot_val=spot_val,
            max_pos_signal=max_pos_signal,
            fastest_signal=fastest_signal,
            max_pos_delta=max_pos_delta,
        )

    candidates = _rank_candidates(candidates, weights=perf_weights)[:limit]
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
        "master_direction": master_direction,
        "gamma_delta_by_strike": {float(k): float(v) for k, v in delta.nlargest(8).items()},
    }


def compute_entry_candidates(
    exposure: pd.Series | None,
    previous: pd.Series | None,
    *,
    spot: float | None,
    ticker: str | None = None,
) -> dict[str, Any]:
    """Return up to N near-spot positive-gamma entry candidates per bar."""
    cur = _clean(exposure)
    prev = _clean(previous)
    spot_val = float(spot or 0.0)
    perf_weights = signal_performance_weights(ticker)

    if cur.empty:
        return {"available": False, "reason": "No gamma exposure data"}

    search = cur
    if spot_val > 0:
        near = select_atm_strike_series(cur, spot_val, window_pct=0.12, min_strikes=5)
        if not near.empty:
            search = near

    positive = search[search > 0]
    negative = search[search < 0]
    max_pos_strike = float(positive.idxmax()) if not positive.empty else float(cur.idxmax())
    max_pos_gamma = float(positive.max()) if not positive.empty else float(cur.max())
    min_neg_strike = float(negative.idxmin()) if not negative.empty else max_pos_strike
    min_neg_gamma = float(negative.min()) if not negative.empty else 0.0
    max_pos_delta = 0.0
    min_neg_delta = 0.0

    delta = cur.subtract(prev.reindex(cur.index), fill_value=0.0) if not prev.empty else cur * 0.0
    fastest_strike = float(delta.idxmax()) if not delta.empty else max_pos_strike
    fastest_delta = float(delta.max()) if not delta.empty else 0.0
    fastest_gamma = float(cur.get(fastest_strike, 0.0))
    if not positive.empty:
        max_pos_delta = float(delta.get(max_pos_strike, 0.0))
    if not negative.empty:
        min_neg_delta = float(delta.get(min_neg_strike, 0.0))

    max_option_type = _option_type_for_strike(max_pos_strike, spot_val)
    max_trade_strike = _resolve_trade_strike(cur, max_pos_strike, spot_val, max_option_type)

    max_pos_signal = GammaSignal(
        signal_type="max_positive_gamma",
        strike=max_trade_strike or max_pos_strike,
        gamma_bn=max_pos_gamma,
        gamma_delta=max_pos_delta,
        score=max_pos_gamma,
        option_type=max_option_type,
        rationale=f"Largest positive gamma magnet {max_pos_strike:.0f}",
        magnet_strike=max_pos_strike,
    )

    min_neg_signal: GammaSignal | None = None
    if not negative.empty:
        min_option_type = _option_type_for_strike(min_neg_strike, spot_val)
        min_neg_signal = GammaSignal(
            signal_type="min_negative_gamma",
            strike=min_neg_strike,
            gamma_bn=min_neg_gamma,
            gamma_delta=min_neg_delta,
            score=abs(min_neg_gamma),
            option_type=min_option_type,
            rationale=f"Largest negative gamma magnet {min_neg_strike:.0f}",
            magnet_strike=min_neg_strike,
        )

    fast_option_type = _option_type_for_strike(fastest_strike, spot_val)
    fast_trade_strike = _resolve_trade_strike(cur, fastest_strike, spot_val, fast_option_type)
    fastest_signal = GammaSignal(
        signal_type="fastest_gamma_increase",
        strike=fast_trade_strike or fastest_strike,
        gamma_bn=fastest_gamma,
        gamma_delta=fastest_delta,
        score=fastest_delta + max(fastest_gamma, 0.0) * 0.25,
        option_type=fast_option_type,
        rationale=f"Fastest gamma increase at {fastest_strike:.0f} (Δ{fastest_delta:+.3f} Bn)",
        magnet_strike=fastest_strike,
    )

    if max_gamma_only():
        result = _compute_max_gamma_only(
            cur=cur,
            search=search,
            spot_val=spot_val,
            delta=delta,
            max_pos_signal=max_pos_signal,
            min_neg_signal=min_neg_signal,
            fastest_signal=fastest_signal,
            max_pos_delta=max_pos_delta,
            perf_weights=perf_weights,
        )
        if result.get("available"):
            delta_slice = delta.nlargest(8)
            result["gamma_delta_by_strike"] = {float(k): float(v) for k, v in delta_slice.items()}
            if perf_weights and result.get("recommended"):
                result["recommended"] = _apply_performance_weight(result["recommended"], perf_weights)
        return result

    return _compute_legacy_candidates(
        cur=cur,
        positive=positive,
        spot_val=spot_val,
        max_pos_strike=max_pos_strike,
        max_pos_delta=max_pos_delta,
        max_pos_signal=max_pos_signal,
        fastest_signal=fastest_signal,
        fastest_strike=fastest_strike,
        fastest_delta=fastest_delta,
        fast_trade_strike=fast_trade_strike,
        delta=delta,
        perf_weights=perf_weights,
    )


def compute_gamma_signals(
    exposure: pd.Series | None,
    previous: pd.Series | None,
    *,
    spot: float | None,
    ticker: str | None = None,
) -> dict[str, Any]:
    """Primary gamma signal bundle (first entry candidate)."""
    return compute_entry_candidates(exposure, previous, spot=spot, ticker=ticker)
