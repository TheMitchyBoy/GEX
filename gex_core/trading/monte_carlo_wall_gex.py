"""Monte Carlo / grid search for wall GEX stop-loss and take-profit."""

from __future__ import annotations

import itertools
import random
from typing import Any, Literal

from gex_core.exports import EXPORT_DIR
from gex_core.history import _build_history_impl
from gex_core.trading.low_gex_backtest import backtest_wall_gex_trader
from gex_core.trading.monte_carlo_search import score_result

WallTarget = Literal["min", "max"]


def _trial(
    *,
    ticker: str,
    history: list[dict],
    starting_capital: float,
    stop_loss: float,
    take_profit: float,
    target: WallTarget,
) -> dict[str, Any]:
    result = backtest_wall_gex_trader(
        ticker,
        target=target,
        history=history,
        stop_loss=stop_loss,
        take_profit=take_profit,
        starting_capital=starting_capital,
        reenter_each_bar=False,
        dedupe_identical_strikes=False,
    )
    account = result.get("account") or {}
    by_exit = result.get("by_exit_reason") or {}
    session_gap = int(by_exit.get("session_gap") or 0)
    total_trades = int(result.get("total_trades") or 0)
    return {
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "target": target,
        "score": score_result(result),
        "total_trades": total_trades,
        "meaningful_trades": max(0, total_trades - session_gap),
        "win_rate": result.get("win_rate"),
        "total_pnl_usd": result.get("total_pnl_usd", 0.0),
        "return_pct": account.get("return_pct", 0.0),
        "max_drawdown_pct": account.get("max_drawdown_pct", 0.0),
        "ending_capital": account.get("ending_capital"),
        "by_exit_reason": by_exit,
    }


def _default_stop_grid() -> list[float]:
    return [round(x, 2) for x in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]]


def _default_take_profit_grid() -> list[float]:
    return [round(x, 2) for x in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]]


def _random_pairs(rng: random.Random, n: int) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for _ in range(n):
        sl = round(rng.uniform(0.02, 0.18), 3)
        tp = round(rng.uniform(0.10, 0.90), 3)
        if tp <= sl:
            tp = round(sl + rng.uniform(0.05, 0.40), 3)
        pairs.append((sl, tp))
    return pairs


def run_wall_gex_monte_carlo(
    *,
    ticker: str = "SPX",
    lookback_days: int = 7,
    starting_capital: float = 500.0,
    target: WallTarget = "min",
    mode: Literal["grid", "random", "both"] = "both",
    random_trials: int = 100,
    seed: int = 42,
    stop_grid: list[float] | None = None,
    take_profit_grid: list[float] | None = None,
    max_snapshots: int = 5000,
) -> dict[str, Any]:
    """Sweep stop-loss / take-profit for wall GEX backtest."""
    ticker = ticker.upper()
    history = _build_history_impl(
        ticker,
        EXPORT_DIR,
        lookback_days=lookback_days,
        max_snapshots=max_snapshots,
        dedupe_identical_strikes=False,
    )
    if len(history) < 2:
        return {
            "ticker": ticker,
            "message": "Not enough export history",
            "snapshots": len(history),
            "trials": [],
        }

    pairs: list[tuple[float, float]] = []
    if mode in {"grid", "both"}:
        sl_grid = stop_grid or _default_stop_grid()
        tp_grid = take_profit_grid or _default_take_profit_grid()
        pairs.extend((sl, tp) for sl, tp in itertools.product(sl_grid, tp_grid) if tp > sl)

    if mode in {"random", "both"}:
        rng = random.Random(seed)
        pairs.extend(_random_pairs(rng, random_trials))

    # Deduplicate while preserving order
    seen: set[tuple[float, float]] = set()
    unique_pairs: list[tuple[float, float]] = []
    for pair in pairs:
        key = (round(pair[0], 4), round(pair[1], 4))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    trials = [
        _trial(
            ticker=ticker,
            history=history,
            starting_capital=starting_capital,
            stop_loss=sl,
            take_profit=tp,
            target=target,
        )
        for sl, tp in unique_pairs
    ]
    ranked = sorted(trials, key=lambda row: row["score"], reverse=True)
    with_trades = [r for r in ranked if r["total_trades"] > 0]
    profitable = [r for r in ranked if float(r.get("total_pnl_usd") or 0) > 0]

    return {
        "ticker": ticker,
        "target": target,
        "lookback_days": lookback_days,
        "starting_capital": starting_capital,
        "seed": seed,
        "mode": mode,
        "snapshots": len(history),
        "date_from": history[0]["ts"],
        "date_to": history[-1]["ts"],
        "trials_run": len(trials),
        "trials_with_trades": len(with_trades),
        "trials_profitable": len(profitable),
        "best": ranked[0] if ranked else None,
        "best_profitable": profitable[0] if profitable else None,
        "top": ranked[: min(15, len(ranked))],
        "all": ranked,
    }
