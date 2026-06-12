#!/usr/bin/env python3
"""Compare backtest: lowest vs highest GEX wall (min vs max γ strike)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.env_bootstrap import load_env_files

load_env_files()
os.environ.setdefault("GEX_WALL_SIGNAL_FILTERS", "0")

from gex_core.trading.config import DEFAULT_WALL_WINDOW_PCT, wall_gex_profile
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
    parser = argparse.ArgumentParser(
        description="Compare min vs max GEX wall strategies on the same snapshot history",
    )
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--starting-capital", type=float, default=500.0)
    parser.add_argument(
        "--window-pct",
        type=float,
        default=DEFAULT_WALL_WINDOW_PCT,
        help="Strike search window as fraction of spot (default 12%%; /near uses 1%%)",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop loss fraction (default: profile for --window-pct)",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Take profit fraction (default: profile for --window-pct)",
    )
    parser.add_argument(
        "--max-hold-bars",
        type=int,
        default=None,
        help="Max hold in bars (default: profile for --window-pct)",
    )
    parser.add_argument("--max-snapshots", type=int, default=5000)
    parser.add_argument(
        "--reenter-each-bar",
        action="store_true",
        help="Rotate position every bar (closes skip stop/target)",
    )
    parser.add_argument("--no-reenter", action="store_true", help="Hold until stop/target (default)")
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Collapse consecutive identical strike profiles (off by default; not recommended for /near)",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep every snapshot (default for near-wall ±1%%)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    reenter = args.reenter_each_bar and not args.no_reenter
    if args.dedupe and args.no_dedupe:
        parser.error("Use only one of --dedupe or --no-dedupe")
    dedupe: bool | None
    if args.dedupe:
        dedupe = True
    elif args.no_dedupe:
        dedupe = False
    else:
        dedupe = None

    profile = wall_gex_profile(args.window_pct)
    result = compare_wall_gex_backtest(
        args.ticker.upper(),
        lookback_days=args.lookback_days,
        starting_capital=args.starting_capital,
        reenter_each_bar=reenter,
        window_pct=args.window_pct,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        max_hold_bars=args.max_hold_bars,
        max_snapshots=args.max_snapshots,
        dedupe_identical_strikes=dedupe,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    c = result["comparison"]
    low = result["low_gex"]
    high = result["high_gex"]
    sl = result.get("stop_loss_pct", profile.stop_loss_pct)
    tp = result.get("take_profit_pct", profile.take_profit_pct)

    print(f"\n=== Wall GEX comparison · {result['ticker']} ({result['lookback_days']}d) ===")
    near = result.get("near_wall")
    profile_label = "near-wall profile" if near else "full-wall profile"
    print(f"strike window: ±{result['window_pct'] * 100:.1f}% of spot ({profile_label})")
    print(f"window: {result.get('date_from')} -> {result.get('date_to')}")
    print(
        f"snapshots: {result.get('snapshots')}  |  dedupe: {result.get('dedupe_identical_strikes')}  |  "
        f"re-enter each bar: {result.get('reenter_each_bar')}"
    )
    print(
        f"stop / target / hold: {-sl:.0%} / {tp:.0%} / {result.get('max_hold_bars')} bars  |  "
        f"shift re-entry: {'on' if result.get('reenter_on_shift') else 'off'}"
    )
    rec = result.get("recommended_side")
    if rec == "min":
        print("recommendation: trade LOW (min γ) wall only")
    elif rec == "max":
        print("recommendation: trade HIGH (max γ) wall only")
    elif rec == "tie":
        print("recommendation: tie — either side similar on this window")
    print()
    print(f"{'':24} {'LOWEST GEX':>14} {'HIGHEST GEX':>14}")
    print("-" * 54)
    print(f"{'Trades':24} {c['low_trades']:>14} {c['high_trades']:>14}")
    print(f"{'Win rate':24} {_fmt_pct(c['low_win_rate']):>14} {_fmt_pct(c['high_win_rate']):>14}")
    print(f"{'Total PnL':24} {_fmt_usd(c['low_pnl_usd']):>14} {_fmt_usd(c['high_pnl_usd']):>14}")
    print(f"{'Account return':24} {_fmt_pct(c['low_return_pct']):>14} {_fmt_pct(c['high_return_pct']):>14}")
    print(f"{'Max drawdown':24} {_fmt_pct(c['low_max_dd']):>14} {_fmt_pct(c['high_max_dd']):>14}")
    print(f"{'Skipped (too far)':24} {c['low_skipped_distance']:>14} {c['high_skipped_distance']:>14}")

    for label, side in (("LOW", low), ("HIGH", high)):
        by = side.get("by_exit_reason") or {}
        if by:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(by.items()))
            print(f"\n{label} exits: {parts}")

    if low.get("total_trades"):
        last_low = low["trades"][-1]
        print(
            f"\nLast LOW trade:  {last_low['option_type']} @ {last_low['strike']:.2f}  "
            f"{_fmt_pct(last_low['pnl_pct'])}"
        )
    if high.get("total_trades"):
        last_high = high["trades"][-1]
        print(f"Last HIGH trade: {last_high['option_type']} @ {last_high['strike']:.2f}  {_fmt_pct(last_high['pnl_pct'])}")


if __name__ == "__main__":
    main()
