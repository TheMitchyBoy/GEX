"""Monte Carlo search over gamma auto-trader configs on export history."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.exports import EXPORT_DIR
from gex_core.history import _build_history_impl
from gex_core.trading.backtest import backtest_auto_trader

# Env keys the sampler may set (restored after each trial).
_TRADER_ENV_KEYS = (
    "GEX_TRADER_STRICT_FILTERS",
    "GEX_TRADER_MIN_GAMMA_DELTA",
    "GEX_TRADER_MIN_FASTEST_GAMMA_DELTA",
    "GEX_TRADER_MIN_AI_CONFIDENCE",
    "GEX_TRADER_STOP_LOSS_PCT",
    "GEX_TRADER_TAKE_PROFIT_PCT",
    "GEX_TRADER_PARTIAL_TP_PCT",
    "GEX_TRADER_TIME_STOP_BARS",
    "GEX_TRADER_STOP_COOLDOWN_BARS",
    "GEX_TRADER_MAX_OPEN",
    "GEX_TRADER_REQUIRE_MOMENTUM",
    "GEX_TRADER_REQUIRE_FLOW_ALIGN",
    "GEX_TRADER_REQUIRE_FLIP_SIDE",
    "GEX_TRADER_ENTRY_TIME_FILTER",
    "GEX_TRADER_MIN_ZERO_DTE_RATIO",
    "GEX_TRADER_MAX_IV_RANK",
    "GEX_TRADER_MOMENTUM_BARS",
    "GEX_TRADER_MIN_MAGNET_PROGRESS",
    "GEX_TRADER_MIN_FLOW_BUY_RATIO",
    "GEX_TRADER_BLOCK_EVENTS",
    "GEX_TRADER_EVENT_SIZE_MULT",
    "GEX_TRADER_RISK_SIZING",
    "GEX_TRADER_RISK_PER_TRADE_PCT",
    "GEX_TRADER_MAGNET_TOUCH_EXIT",
    "GEX_TRADER_EOD_FLATTEN",
    "GEX_TRADER_BAR_MINUTES",
)


@dataclass(frozen=True)
class TraderConfig:
    name: str
    env: dict[str, str] = field(default_factory=dict)
    stop_loss: float | None = None
    take_profit: float | None = None
    min_confidence: float | None = None
    max_open: int | None = None


def _baseline_configs() -> list[TraderConfig]:
    """Fixed presets including current production defaults."""
    return [
        TraderConfig(name="production"),
        TraderConfig(
            name="relaxed_filters",
            env={
                "GEX_TRADER_STRICT_FILTERS": "0",
                "GEX_TRADER_ENTRY_TIME_FILTER": "0",
            },
        ),
        TraderConfig(
            name="scalp_loose",
            env={
                "GEX_TRADER_STRICT_FILTERS": "0",
                "GEX_TRADER_ENTRY_TIME_FILTER": "0",
                "GEX_TRADER_MIN_GAMMA_DELTA": "0.02",
                "GEX_TRADER_MIN_FASTEST_GAMMA_DELTA": "0.03",
                "GEX_TRADER_REQUIRE_MOMENTUM": "0",
                "GEX_TRADER_REQUIRE_FLOW_ALIGN": "0",
                "GEX_TRADER_REQUIRE_FLIP_SIDE": "0",
                "GEX_TRADER_MIN_ZERO_DTE_RATIO": "0",
                "GEX_TRADER_STOP_LOSS_PCT": "0.03",
                "GEX_TRADER_TAKE_PROFIT_PCT": "0.10",
                "GEX_TRADER_TIME_STOP_BARS": "2",
                "GEX_TRADER_STOP_COOLDOWN_BARS": "1",
                "GEX_TRADER_MAX_OPEN": "3",
            },
            min_confidence=0.45,
        ),
        TraderConfig(
            name="swing_strict",
            env={
                "GEX_TRADER_STRICT_FILTERS": "1",
                "GEX_TRADER_ENTRY_TIME_FILTER": "1",
                "GEX_TRADER_MIN_GAMMA_DELTA": "0.05",
                "GEX_TRADER_TAKE_PROFIT_PCT": "0.35",
                "GEX_TRADER_TIME_STOP_BARS": "6",
                "GEX_TRADER_MAX_OPEN": "1",
            },
            min_confidence=0.65,
        ),
    ]


def _random_config(rng: random.Random, trial: int) -> TraderConfig:
    strict = rng.choice(["0", "1"])
    env = {
        "GEX_TRADER_STRICT_FILTERS": strict,
        "GEX_TRADER_MIN_GAMMA_DELTA": f"{rng.uniform(0.01, 0.08):.3f}",
        "GEX_TRADER_MIN_FASTEST_GAMMA_DELTA": f"{rng.uniform(0.02, 0.15):.3f}",
        "GEX_TRADER_MIN_AI_CONFIDENCE": f"{rng.uniform(0.35, 0.75):.2f}",
        "GEX_TRADER_STOP_LOSS_PCT": f"{rng.uniform(0.03, 0.08):.3f}",
        "GEX_TRADER_TAKE_PROFIT_PCT": f"{rng.uniform(0.08, 0.40):.2f}",
        "GEX_TRADER_PARTIAL_TP_PCT": f"{rng.uniform(0.05, 0.20):.2f}",
        "GEX_TRADER_TIME_STOP_BARS": str(rng.randint(1, 8)),
        "GEX_TRADER_STOP_COOLDOWN_BARS": str(rng.randint(0, 6)),
        "GEX_TRADER_MAX_OPEN": str(rng.randint(1, 5)),
        "GEX_TRADER_REQUIRE_MOMENTUM": rng.choice(["0", "1"]),
        "GEX_TRADER_REQUIRE_FLOW_ALIGN": rng.choice(["0", "1"]),
        "GEX_TRADER_REQUIRE_FLIP_SIDE": rng.choice(["0", "1"]),
        "GEX_TRADER_ENTRY_TIME_FILTER": rng.choice(["0", "1"]),
        "GEX_TRADER_MIN_ZERO_DTE_RATIO": rng.choice(["0", "0.2", "0.4"]),
        "GEX_TRADER_MAX_IV_RANK": f"{rng.uniform(0.75, 1.0):.2f}",
        "GEX_TRADER_MOMENTUM_BARS": str(rng.randint(1, 3)),
        "GEX_TRADER_MIN_MAGNET_PROGRESS": f"{rng.uniform(0.0, 0.2):.2f}",
        "GEX_TRADER_MIN_FLOW_BUY_RATIO": f"{rng.uniform(0.0, 0.6):.2f}",
        "GEX_TRADER_BLOCK_EVENTS": rng.choice(["0", "1"]),
        "GEX_TRADER_EVENT_SIZE_MULT": rng.choice(["0", "0.5", "1"]),
        "GEX_TRADER_RISK_SIZING": "1",
        "GEX_TRADER_RISK_PER_TRADE_PCT": f"{rng.uniform(0.005, 0.03):.3f}",
        "GEX_TRADER_MAGNET_TOUCH_EXIT": rng.choice(["0", "1"]),
        "GEX_TRADER_EOD_FLATTEN": rng.choice(["0", "1"]),
        "GEX_TRADER_BAR_MINUTES": str(rng.choice([2, 5, 10])),
    }
    if strict == "1":
        env.setdefault("GEX_TRADER_REQUIRE_MOMENTUM", "1")
    return TraderConfig(name=f"random_{trial:04d}", env=env)


@contextmanager
def _trader_env(config: TraderConfig) -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in _TRADER_ENV_KEYS}
    try:
        for key in _TRADER_ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in config.env.items():
            os.environ[key] = value
        yield
    finally:
        for key in _TRADER_ENV_KEYS:
            if saved[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = saved[key]


def score_result(result: dict[str, Any]) -> float:
    """Rank configs: PnL and return first; penalize zero-trade and session-gap-only runs."""
    trades = int(result.get("total_trades") or 0)
    if trades == 0:
        return -1e9

    account = result.get("account") or {}
    ret = float(account.get("return_pct") or result.get("return_pct") or 0.0)
    dd = float(account.get("max_drawdown_pct") or 0.0)
    win_rate = float(result.get("win_rate") or 0.0)
    pnl = float(result.get("total_pnl_usd") or 0.0)

    by_exit = result.get("by_exit_reason") or {}
    session_gap = int(by_exit.get("session_gap") or 0)
    meaningful = trades - session_gap

    # Session-gap-only configs (off-hours snapshot artifacts) rank below flat no-trade runs.
    if meaningful <= 0 and pnl == 0.0 and ret == 0.0:
        return -1e6 + trades

    activity = min(meaningful if meaningful > 0 else trades, 20) / 20.0
    return ret * 1000.0 + pnl * 0.5 + win_rate * 50.0 - dd * 200.0 + activity * 10.0


def run_trial(
    config: TraderConfig,
    *,
    ticker: str,
    history: list[dict],
    starting_capital: float,
) -> dict[str, Any]:
    with _trader_env(config):
        result = backtest_auto_trader(
            ticker,
            history=history,
            starting_capital=starting_capital,
            stop_loss=config.stop_loss,
            take_profit=config.take_profit,
            min_confidence=config.min_confidence,
            max_open=config.max_open,
        )
    account = result.get("account") or {}
    by_exit = result.get("by_exit_reason") or {}
    session_gap = int(by_exit.get("session_gap") or 0)
    total_trades = int(result.get("total_trades") or 0)
    return {
        "name": config.name,
        "score": score_result(result),
        "total_trades": total_trades,
        "meaningful_trades": max(0, total_trades - session_gap),
        "session_gap_trades": session_gap,
        "win_rate": result.get("win_rate"),
        "total_pnl_usd": result.get("total_pnl_usd", 0.0),
        "avg_pnl_pct": result.get("avg_pnl_pct"),
        "return_pct": account.get("return_pct", 0.0),
        "ending_capital": account.get("ending_capital"),
        "max_drawdown_pct": account.get("max_drawdown_pct", 0.0),
        "skipped_gamma_decline": result.get("skipped_gamma_decline", 0),
        "skipped_filters": result.get("skipped_filters", 0),
        "config": {
            "env": dict(config.env),
            "stop_loss": config.stop_loss,
            "take_profit": config.take_profit,
            "min_confidence": config.min_confidence,
            "max_open": config.max_open,
        },
        "by_exit_reason": result.get("by_exit_reason"),
        "date_from": result.get("date_from"),
        "date_to": result.get("date_to"),
    }


def run_monte_carlo(
    *,
    ticker: str,
    lookback_days: int,
    trials: int,
    starting_capital: float,
    seed: int,
    max_snapshots: int = 500,
) -> dict[str, Any]:
    history = _build_history_impl(
        ticker.upper(),
        EXPORT_DIR,
        lookback_days=lookback_days,
        max_snapshots=max_snapshots,
        dedupe_identical_strikes=False,
    )
    if len(history) < 2:
        return {
            "ticker": ticker.upper(),
            "message": "Not enough export history for Monte Carlo",
            "snapshots": len(history),
            "trials": [],
        }

    rng = random.Random(seed)
    configs = _baseline_configs()
    configs.extend(_random_config(rng, i) for i in range(max(0, trials - len(configs))))

    results = [
        run_trial(cfg, ticker=ticker.upper(), history=history, starting_capital=starting_capital)
        for cfg in configs
    ]
    ranked = sorted(results, key=lambda row: row["score"], reverse=True)
    with_trades = [r for r in ranked if r["total_trades"] > 0]
    meaningful = [r for r in ranked if r.get("meaningful_trades", 0) > 0]
    profitable = [r for r in ranked if float(r.get("total_pnl_usd") or 0) > 0]

    return {
        "ticker": ticker.upper(),
        "lookback_days": lookback_days,
        "starting_capital": starting_capital,
        "seed": seed,
        "snapshots": len(history),
        "date_from": history[0]["ts"],
        "date_to": history[-1]["ts"],
        "trials_run": len(results),
        "trials_with_trades": len(with_trades),
        "trials_with_meaningful_trades": len(meaningful),
        "trials_profitable": len(profitable),
        "production": next((r for r in results if r["name"] == "production"), None),
        "best": ranked[0] if ranked else None,
        "best_with_trades": with_trades[0] if with_trades else None,
        "best_meaningful": meaningful[0] if meaningful else None,
        "best_profitable": profitable[0] if profitable else None,
        "top": ranked[: min(10, len(ranked))],
        "all": ranked,
    }


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo search for auto-trader configs")
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--trials", type=int, default=200, help="Random configs (+ baselines)")
    parser.add_argument("--starting-capital", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    summary = run_monte_carlo(
        ticker=args.ticker.upper(),
        lookback_days=args.lookback_days,
        trials=args.trials,
        starting_capital=args.starting_capital,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return

    if summary.get("message"):
        print(summary["message"])
        return

    print(f"\n=== Monte Carlo trader configs: {summary['ticker']} ===")
    print(f"window: {summary['date_from']} -> {summary['date_to']} ({summary['snapshots']} snapshots)")
    print(f"trials: {summary['trials_run']} ({summary['trials_with_trades']} produced trades)")
    print(
        f"meaningful trades: {summary['trials_with_meaningful_trades']} | "
        f"profitable: {summary['trials_profitable']}"
    )
    print(f"starting capital: ${summary['starting_capital']:,.2f}")

    prod = summary.get("production")
    if prod:
        print(
            f"\nProduction baseline: {prod['total_trades']} trades "
            f"({prod.get('meaningful_trades', 0)} meaningful) | "
            f"return {_fmt_pct(prod.get('return_pct'))} | PnL ${prod.get('total_pnl_usd', 0):,.2f}"
        )

    best = summary.get("best_profitable") or summary.get("best_meaningful") or summary.get("best_with_trades") or summary.get("best")
    if not best:
        print("No results.")
        return

    if summary["trials_with_trades"] == 0:
        print("\nNo config produced trades in this window.")
        print("Top scored (all zero-trade):")
    elif summary["trials_profitable"] == 0 and summary["trials_with_meaningful_trades"] == 0:
        print("\nNo profitable or intraday-quality trades in this window.")
        print("Best among off-hours session-gap configs (flat $0 PnL):")
    else:
        print("\nBest config (by score):")

    print(f"  name: {best['name']}")
    print(f"  trades: {best['total_trades']} ({best.get('meaningful_trades', 0)} meaningful) | win rate: {_fmt_pct(best.get('win_rate'))}")
    print(f"  PnL: ${best.get('total_pnl_usd', 0):,.2f} | return: {_fmt_pct(best.get('return_pct'))}")
    print(f"  ending: ${best.get('ending_capital', summary['starting_capital']):,.2f} | max DD: {_fmt_pct(best.get('max_drawdown_pct'))}")
    if best.get("by_exit_reason"):
        print(f"  exits: {best['by_exit_reason']}")
    env = best["config"].get("env") or {}
    if env:
        print("  env overrides:")
        for key in sorted(env):
            print(f"    {key}={env[key]}")

    print(f"\nTop {args.top}:")
    for row in summary["top"][: args.top]:
        print(
            f"  {row['name']:16s} score={row['score']:8.1f} "
            f"trades={row['total_trades']:3d} mean={row.get('meaningful_trades', 0):2d} "
            f"ret={_fmt_pct(row.get('return_pct'))} "
            f"pnl=${row.get('total_pnl_usd', 0):7.2f} wr={_fmt_pct(row.get('win_rate'))}"
        )


if __name__ == "__main__":
    main()
