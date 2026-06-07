"""CLI: walk-forward backtest for the gamma auto-trader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.trading.backtest import backtest_auto_trader, backtest_auto_trader_bootstrap


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest gamma auto-trader on export history")
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--max-snapshots", type=int, default=500)
    parser.add_argument("--dedupe", action="store_true", help="Skip consecutive identical strike profiles")
    parser.add_argument("--json", action="store_true", help="Print full JSON result")
    parser.add_argument("--stop-loss", type=float, default=None)
    parser.add_argument("--take-profit", type=float, default=None)
    parser.add_argument("--target-trades", type=int, default=0, help="Bootstrap until N trades (e.g. 1000)")
    parser.add_argument("--starting-capital", type=float, default=None, help="Simulate account equity (e.g. 500)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.target_trades > 0:
        result = backtest_auto_trader_bootstrap(
            args.ticker.upper(),
            target_trades=args.target_trades,
            seed=args.seed,
            lookback_days=args.lookback_days,
            max_snapshots=args.max_snapshots,
            dedupe_identical_strikes=args.dedupe,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            starting_capital=args.starting_capital,
        )
    else:
        result = backtest_auto_trader(
            args.ticker.upper(),
            lookback_days=args.lookback_days,
            max_snapshots=args.max_snapshots,
            dedupe_identical_strikes=args.dedupe,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            starting_capital=args.starting_capital,
        )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    print(f"\n=== Auto-trader backtest: {result.get('ticker', args.ticker.upper())} ===")
    if result.get("date_from") and result.get("date_to"):
        print(f"window: {result['date_from']} -> {result['date_to']}")
    if result.get("execution_ticker"):
        print(f"execution: {result['execution_ticker']} (gamma signals on {result.get('ticker', args.ticker.upper())})")
    if result.get("message") and not result.get("total_trades"):
        print(result["message"])
        print(f"snapshots: {result.get('snapshots', 0)}")
        if result.get("weekend_snapshots_excluded"):
            print(f"weekend snapshots excluded: {result['weekend_snapshots_excluded']}")
        account = result.get("account")
        if account:
            print(
                f"account: ${account['starting_capital']:,.2f} -> ${account['ending_capital']:,.2f} "
                f"({account['return_pct']:+.1%})"
            )
        return

    print(f"snapshots: {result['snapshots']}")
    if result.get("bootstrap"):
        print(f"base snapshots: {result.get('base_snapshots', 'n/a')}")
        print(f"target trades: {result.get('target_trades', 'n/a')}")
    print(f"total trades: {result['total_trades']}")
    print(f"win rate: {result['win_rate']:.1%} ({result['wins']}W / {result['losses']}L)")
    print(f"avg PnL: {_fmt_pct(result['avg_pnl_pct'])}")
    print(f"total PnL: ${result['total_pnl_usd']:,.2f}")
    print(f"avg bars held: {result['avg_bars_held']:.1f}")
    print(f"skipped entries: {result['skipped_entries']}")
    print(f"blocked duplicates: {result.get('blocked_duplicate', 0)}")
    print(f"skipped gamma decline: {result.get('skipped_gamma_decline', 0)}")
    print(f"skipped strike distance: {result.get('skipped_strike_distance', 0)}")
    print(f"skipped filters: {result.get('skipped_filters', 0)}")
    print(f"blocked cooldown: {result.get('blocked_cooldown', 0)}")
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
        if account.get("skipped_insufficient_capital"):
            print(f"  skipped (insufficient capital): {account['skipped_insufficient_capital']}")

    print("\nBy signal type:")
    for sig, stats in sorted((result.get("by_signal") or {}).items()):
        print(
            f"  {sig}: {stats['count']} trades, "
            f"win {stats['win_rate']:.0%}, avg {_fmt_pct(stats['avg_pnl_pct'])}, "
            f"${stats['avg_pnl_usd']:,.2f}/trade"
        )

    print("\nBy exit reason:")
    for reason, count in sorted((result.get("by_exit_reason") or {}).items()):
        print(f"  {reason}: {count}")

    trades = result.get("trades") or []
    if trades:
        print("\nRecent trades (last 10):")
        for t in trades[-10:]:
            strike_label = f"{t['strike']:.2f}"
            if t.get("signal_strike"):
                strike_label = f"SPY {t['strike']:.2f} (SPX {t['signal_strike']:.0f})"
            qty = t.get("qty", 1)
            equity = t.get("equity_after")
            equity_suffix = f" | bal ${equity:,.2f}" if equity is not None else ""
            print(
                f"  {t['entry_ts']} -> {t['exit_ts']} | {t['signal_type']} "
                f"{t['option_type']} {strike_label} x{qty:g} | {_fmt_pct(t['pnl_pct'])} "
                f"(${t['pnl_usd']:,.2f}) [{t['exit_reason']}]{equity_suffix}"
            )


if __name__ == "__main__":
    main()
