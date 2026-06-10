#!/usr/bin/env python3
"""Monte Carlo search for optimal wall GEX stop-loss and take-profit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.env_bootstrap import load_env_files

load_env_files()

from gex_core.trading.monte_carlo_wall_gex import run_wall_gex_monte_carlo


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo SL/TP search for wall GEX strategy")
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--starting-capital", type=float, default=500.0)
    parser.add_argument("--target", choices=["min", "max"], default="min", help="min=lowest GEX, max=highest")
    parser.add_argument("--mode", choices=["grid", "random", "both"], default="both")
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = run_wall_gex_monte_carlo(
        ticker=args.ticker.upper(),
        lookback_days=args.lookback_days,
        starting_capital=args.starting_capital,
        target=args.target,
        mode=args.mode,
        random_trials=args.random_trials,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return

    if summary.get("message"):
        print(summary["message"])
        return

    label = "LOWEST" if args.target == "min" else "HIGHEST"
    print(f"\n=== Wall GEX Monte Carlo ({label}) · {summary['ticker']} ({summary['lookback_days']}d) ===")
    print(f"window: {summary['date_from']} -> {summary['date_to']} ({summary['snapshots']} snapshots)")
    print(
        f"trials: {summary['trials_run']} | with trades: {summary['trials_with_trades']} | "
        f"profitable: {summary['trials_profitable']}"
    )
    print(f"starting capital: ${summary['starting_capital']:,.2f}")

    best = summary.get("best_profitable") or summary.get("best")
    if not best:
        print("No results.")
        return

    print("\nOptimal (by score):")
    print(f"  stop-loss:   {best['stop_loss']:.1%}")
    print(f"  take-profit: {best['take_profit']:.1%}")
    print(f"  trades: {best['total_trades']} | win rate: {_fmt_pct(best.get('win_rate'))}")
    print(f"  PnL: ${best.get('total_pnl_usd', 0):,.2f} | return: {_fmt_pct(best.get('return_pct'))}")
    print(f"  max DD: {_fmt_pct(best.get('max_drawdown_pct'))}")
    if best.get("by_exit_reason"):
        print(f"  exits: {best['by_exit_reason']}")

    print(f"\nTop {args.top} (stop / target / return / PnL / trades):")
    for row in summary["top"][: args.top]:
        print(
            f"  SL {row['stop_loss']:5.1%}  TP {row['take_profit']:5.1%}  "
            f"ret={_fmt_pct(row.get('return_pct')):>7}  "
            f"pnl=${row.get('total_pnl_usd', 0):7.2f}  "
            f"trades={row['total_trades']:3d}  "
            f"DD={_fmt_pct(row.get('max_drawdown_pct')):>7}  "
            f"score={row['score']:7.1f}"
        )


if __name__ == "__main__":
    main()
