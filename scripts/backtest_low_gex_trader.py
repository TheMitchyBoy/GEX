#!/usr/bin/env python3
"""CLI: walk-forward backtest for the low-GEX strike trader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.trading.low_gex_backtest import backtest_low_gex_trader


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest low-GEX wall strategy (call/put toward minimum gamma strike)",
    )
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--max-snapshots", type=int, default=500)
    parser.add_argument("--dedupe", action="store_true", help="Skip identical consecutive strike profiles")
    parser.add_argument("--json", action="store_true", help="Print full JSON result")
    parser.add_argument("--stop-loss", type=float, default=0.03, help="Stop loss fraction (default 3%%)")
    parser.add_argument("--take-profit", type=float, default=0.20, help="Take profit fraction (default 20%%)")
    parser.add_argument("--starting-capital", type=float, default=500.0)
    parser.add_argument(
        "--reenter-each-bar",
        action="store_true",
        help="Close prior position every bar and open fresh toward current lowest GEX",
    )
    args = parser.parse_args()

    result = backtest_low_gex_trader(
        args.ticker.upper(),
        lookback_days=args.lookback_days,
        max_snapshots=args.max_snapshots,
        dedupe_identical_strikes=args.dedupe,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        starting_capital=args.starting_capital,
        reenter_each_bar=args.reenter_each_bar,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    print(f"\n=== Low-GEX backtest · {result.get('ticker', args.ticker.upper())} ===")
    if result.get("reenter_each_bar"):
        print("mode: re-enter each bar (close + open toward lowest GEX)")
    if result.get("date_from") and result.get("date_to"):
        print(f"window: {result['date_from']} -> {result['date_to']}")
    if result.get("execution_ticker"):
        print(f"execution: {result['execution_ticker']}")

    if result.get("message") and not result.get("total_trades"):
        print(result["message"])
        print(f"snapshots: {result.get('snapshots', 0)}")
        account = result.get("account")
        if account:
            print(f"account: ${account['starting_capital']:,.2f} -> ${account['ending_capital']:,.2f}")
        return

    print(f"snapshots: {result['snapshots']}")
    print(f"total trades: {result['total_trades']}")
    print(f"win rate: {result['win_rate']:.1%} ({result['wins']}W / {result['losses']}L)")
    print(f"avg PnL: {_fmt_pct(result['avg_pnl_pct'])}")
    print(f"total PnL: ${result['total_pnl_usd']:,.2f}")
    print(f"avg bars held: {result['avg_bars_held']:.1f}")
    print(f"skipped entries: {result['skipped_entries']}")
    print(f"blocked duplicates: {result.get('blocked_duplicate', 0)}")
    print(f"skipped strike distance: {result.get('skipped_strike_distance', 0)}")
    if result.get("off_hours_snapshots_excluded"):
        print(f"off-hours snapshots excluded: {result['off_hours_snapshots_excluded']}")
    if result.get("intraday_session"):
        print("intraday session filter: on")
    if result.get("entry_time_filter"):
        print(f"entry-time filter: on (skipped {result.get('skipped_filters', 0)})")
    if result.get("weekend_snapshots_excluded"):
        print(f"weekend snapshots excluded: {result['weekend_snapshots_excluded']}")
    print(f"stop / target: {_fmt_pct(-result['stop_loss_pct'])} / {_fmt_pct(result['take_profit_pct'])}")

    account = result.get("account")
    if account:
        print("\nAccount simulation:")
        print(f"  starting: ${account['starting_capital']:,.2f}")
        print(f"  ending:   ${account['ending_capital']:,.2f}")
        print(f"  return:   {account['return_pct']:+.1%}")
        print(f"  max drawdown: {account['max_drawdown_pct']:.1%}")

    print("\nBy exit reason:")
    for reason, count in sorted((result.get("by_exit_reason") or {}).items()):
        print(f"  {reason}: {count}")

    trades = result.get("trades") or []
    if trades:
        print(f"\nAll trades ({len(trades)}):")
        for t in trades:
            strike_label = f"{t['strike']:.2f}"
            if t.get("signal_strike"):
                strike_label = f"SPY {t['strike']:.2f} (SPX {t['signal_strike']:.0f})"
            print(
                f"  {t['entry_ts']} -> {t['exit_ts']} | {t['option_type']} {strike_label} "
                f"x{t.get('qty', 1):g} | {_fmt_pct(t['pnl_pct'])} (${t['pnl_usd']:,.2f}) [{t['exit_reason']}]"
            )


if __name__ == "__main__":
    main()
