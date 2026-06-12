"""Monte Carlo / grid search for near-spot (±1%) wall GEX parameters."""

from __future__ import annotations

import itertools
from typing import Any

from gex_core.exports import EXPORT_DIR
from gex_core.history import _build_history_impl
from gex_core.trading.low_gex_backtest import backtest_wall_gex_trader
from gex_core.trading.monte_carlo_search import score_result

NEAR_WINDOW_PCT = 0.01


def _run_trial(
    *,
    ticker: str,
    history: list[dict],
    starting_capital: float,
    stop_loss: float,
    take_profit: float,
    max_hold_bars: int,
    window_pct: float = NEAR_WINDOW_PCT,
    signal_filters: bool = True,
    min_gamma_bn: float = 0.0,
    min_entry_drift_pts: float = 0.0,
    wall_shift_min_pts: float = 0.5,
    wall_shift_cooldown_bars: int = 0,
    reenter_on_shift: bool = True,
) -> dict[str, Any]:
    result = backtest_wall_gex_trader(
        ticker,
        target="min",
        history=history,
        window_pct=window_pct,
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_hold_bars=max_hold_bars,
        starting_capital=starting_capital,
        reenter_each_bar=False,
        reenter_on_shift=reenter_on_shift,
        dedupe_identical_strikes=False,
        signal_filters=signal_filters,
        min_gamma_bn=min_gamma_bn if signal_filters else None,
        min_entry_drift_pts=min_entry_drift_pts if signal_filters else None,
        wall_shift_min_pts=wall_shift_min_pts,
        wall_shift_cooldown_bars=wall_shift_cooldown_bars,
        intraday_session=True,
        entry_time_filter=True,
    )
    account = result.get("account") or {}
    by_exit = result.get("by_exit_reason") or {}
    return {
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "max_hold_bars": max_hold_bars,
        "window_pct": window_pct,
        "signal_filters": signal_filters,
        "min_gamma_bn": min_gamma_bn,
        "min_entry_drift_pts": min_entry_drift_pts,
        "wall_shift_min_pts": wall_shift_min_pts,
        "wall_shift_cooldown_bars": wall_shift_cooldown_bars,
        "reenter_on_shift": reenter_on_shift,
        "score": score_result(result),
        "total_trades": int(result.get("total_trades") or 0),
        "win_rate": result.get("win_rate"),
        "total_pnl_usd": result.get("total_pnl_usd", 0.0),
        "return_pct": account.get("return_pct", 0.0),
        "max_drawdown_pct": account.get("max_drawdown_pct", 0.0),
        "by_exit_reason": by_exit,
        "blocked_duplicate": result.get("blocked_duplicate", 0),
        "skipped_wall_shift_cooldown": result.get("skipped_wall_shift_cooldown", 0),
    }


def run_near_wall_sl_tp_mc(
    *,
    ticker: str = "SPX",
    lookback_days: int = 14,
    starting_capital: float = 500.0,
    max_snapshots: int = 5000,
    window_pct: float = NEAR_WINDOW_PCT,
) -> dict[str, Any]:
    """Grid SL / TP / max_hold for near-spot wall GEX (baseline shift settings)."""
    history = _build_history_impl(
        ticker.upper(),
        EXPORT_DIR,
        lookback_days=lookback_days,
        max_snapshots=max_snapshots,
        dedupe_identical_strikes=False,
    )
    if len(history) < 2:
        return {"message": "Not enough history", "snapshots": len(history), "trials": []}

    stop_grid = [0.03, 0.04, 0.05, 0.06, 0.08]
    tp_grid = [0.15, 0.18, 0.22, 0.28, 0.35]
    hold_grid = [4, 6, 8, 10]

    trials = [
        _run_trial(
            ticker=ticker.upper(),
            history=history,
            starting_capital=starting_capital,
            stop_loss=sl,
            take_profit=tp,
            max_hold_bars=hold,
            window_pct=window_pct,
            signal_filters=False,
            wall_shift_min_pts=0.5,
        )
        for sl, tp, hold in itertools.product(stop_grid, tp_grid, hold_grid)
        if tp > sl
    ]
    ranked = sorted(trials, key=lambda r: r["score"], reverse=True)
    profitable = [r for r in ranked if float(r.get("total_pnl_usd") or 0) > 0 and r["total_trades"] > 0]
    return {
        "phase": "sl_tp_hold",
        "ticker": ticker.upper(),
        "window_pct": window_pct,
        "lookback_days": lookback_days,
        "snapshots": len(history),
        "date_from": history[0]["ts"],
        "date_to": history[-1]["ts"],
        "trials_run": len(trials),
        "best": ranked[0] if ranked else None,
        "best_profitable": profitable[0] if profitable else None,
        "top": ranked[:20],
    }


def run_near_wall_shift_sweep(
    *,
    ticker: str = "SPX",
    lookback_days: int = 14,
    starting_capital: float = 500.0,
    max_snapshots: int = 5000,
    window_pct: float = NEAR_WINDOW_PCT,
    stop_loss: float,
    take_profit: float,
    max_hold_bars: int,
) -> dict[str, Any]:
    """Sweep wall-shift anti-flicker settings using fixed SL/TP/hold from phase 1."""
    history = _build_history_impl(
        ticker.upper(),
        EXPORT_DIR,
        lookback_days=lookback_days,
        max_snapshots=max_snapshots,
        dedupe_identical_strikes=False,
    )
    if len(history) < 2:
        return {"message": "Not enough history", "trials": []}

    min_gamma_grid = [0.0, 0.5, 0.8, 1.0]
    entry_drift_grid = [0.0, 10.0, 15.0, 20.0]
    shift_min_grid = [0.5, 10.0, 15.0, 20.0, 25.0]
    shift_cooldown_grid = [0, 1, 2]

    trials: list[dict[str, Any]] = []
    for min_g, drift, shift_min, cooldown in itertools.product(
        min_gamma_grid,
        entry_drift_grid,
        shift_min_grid,
        shift_cooldown_grid,
    ):
        trials.append(
            _run_trial(
                ticker=ticker.upper(),
                history=history,
                starting_capital=starting_capital,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_hold_bars=max_hold_bars,
                window_pct=window_pct,
                signal_filters=True,
                min_gamma_bn=min_g,
                min_entry_drift_pts=drift,
                wall_shift_min_pts=shift_min,
                wall_shift_cooldown_bars=cooldown,
            )
        )

    ranked = sorted(trials, key=lambda r: r["score"], reverse=True)
    profitable = [r for r in ranked if float(r.get("total_pnl_usd") or 0) > 0 and r["total_trades"] > 0]
    return {
        "phase": "wall_shift",
        "fixed": {"stop_loss": stop_loss, "take_profit": take_profit, "max_hold_bars": max_hold_bars},
        "trials_run": len(trials),
        "best": ranked[0] if ranked else None,
        "best_profitable": profitable[0] if profitable else None,
        "top": ranked[:20],
        "baseline_no_filters": _run_trial(
            ticker=ticker.upper(),
            history=history,
            starting_capital=starting_capital,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold_bars=max_hold_bars,
            window_pct=window_pct,
            signal_filters=False,
            wall_shift_min_pts=0.5,
        ),
    }


def run_near_wall_full_search(
    *,
    ticker: str = "SPX",
    lookback_days: int = 14,
    starting_capital: float = 500.0,
) -> dict[str, Any]:
    phase1 = run_near_wall_sl_tp_mc(
        ticker=ticker,
        lookback_days=lookback_days,
        starting_capital=starting_capital,
    )
    best = phase1.get("best_profitable") or phase1.get("best")
    if not best or not best.get("total_trades"):
        return {"phase1": phase1, "phase2": None}

    phase2 = run_near_wall_shift_sweep(
        ticker=ticker,
        lookback_days=lookback_days,
        starting_capital=starting_capital,
        stop_loss=float(best["stop_loss"]),
        take_profit=float(best["take_profit"]),
        max_hold_bars=int(best["max_hold_bars"]),
    )
    return {"phase1": phase1, "phase2": phase2}
