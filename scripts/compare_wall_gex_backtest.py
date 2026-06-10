#!/usr/bin/env python3
"""Compare 1-week backtest: lowest vs highest spot-exposures/strike GEX wall."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.env_bootstrap import load_env_files

load_env_files()

from gex_core.trading.low_gex_backtest import compare_wall_gex_backtest


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.1%}"


def _fmt_usd(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"${v:,.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare min vs max GEX wall strategies")
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--starting-capital", type=float, default=500.0)
    parser.add_argument("--stop-loss", type=float, default=0.03, help="Stop loss fraction (default 3%%)")
    parser.add_argument("--take-profit", type=float, default=0.20, help="Take profit fraction (default 20%%)")
    parser.add_argument(
        "--reenter-each-bar",
        action="store_true",
        help="Rotate position every bar (closes skip stop/target)",
    )
    parser.add_argument("--no-reenter", action="store_true", help="Hold until stop/target (default)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Collapse consecutive identical strike profiles (off by default for bar rotation)",
    )
    args = parser.parse_args()

    reenter = args.reenter_each_bar and not args.no_reenter
    result = compare_wall_gex_backtest(
        args.ticker.upper(),
        lookback_days=args.lookback_days,
        starting_capital=args.starting_capital,
        reenter_each_bar=reenter,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        dedupe_identical_strikes=args.dedupe,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    c = result["comparison"]
    low = result["low_gex"]
    high = result["high_gex"]

    print(f"\n=== Wall GEX comparison · {result['ticker']} ({result['lookback_days']}d) ===")
    print(f"window: {result.get('date_from')} -> {result.get('date_to')}")
    print(
        f"snapshots: {result.get('snapshots')}  |  re-enter each bar: {result.get('reenter_each_bar')}  |  "
        f"stop/target: {-args.stop_loss:.0%} / {args.take_profit:.0%}"
    )
    print()
    print(f"{'':24} {'LOWEST GEX':>14} {'HIGHEST GEX':>14}")
    print("-" * 54)
    print(f"{'Trades':24} {c['low_trades']:>14} {c['high_trades']:>14}")
    print(f"{'Win rate':24} {_fmt_pct(c['low_win_rate']):>14} {_fmt_pct(c['high_win_rate']):>14}")
    print(f"{'Total PnL':24} {_fmt_usd(c['low_pnl_usd']):>14} {_fmt_usd(c['high_pnl_usd']):>14}")
    print(f"{'Account return':24} {_fmt_pct(c['low_return_pct']):>14} {_fmt_pct(c['high_return_pct']):>14}")
    print(f"{'Max drawdown':24} {_fmt_pct(c['low_max_dd']):>14} {_fmt_pct(c['high_max_dd']):>14}")
    print(f"{'Skipped (too far)':24} {c['low_skipped_distance']:>14} {c['high_skipped_distance']:>14}")

    if low.get("total_trades"):
        last_low = low["trades"][-1]
        print(f"\nLast LOW trade:  {last_low['option_type']} @ {last_low['strike']:.2f}  {_fmt_pct(last_low['pnl_pct'])}")
    if high.get("total_trades"):
        last_high = high["trades"][-1]
        print(f"Last HIGH trade: {last_high['option_type']} @ {last_high['strike']:.2f}  {_fmt_pct(last_high['pnl_pct'])}")


if __name__ == "__main__":
    main()
