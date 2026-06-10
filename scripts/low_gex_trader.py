#!/usr/bin/env python3
"""Buy calls or puts toward the lowest net GEX strike (UW spot-exposures/strike).

Examples
--------
Signal only (default)::

    python scripts/low_gex_trader.py --ticker SPX

Paper trade (opens journal position)::

    python scripts/low_gex_trader.py --ticker SPX --execute

Live Webull (requires GEX_TRADER_PAPER=0 + Webull credentials)::

    python scripts/low_gex_trader.py --ticker SPX --execute --live
"""

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

from gex_core.trading.low_gex_engine import fetch_gex_exposure, run_low_gex_trade


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trade toward the lowest net gamma strike (call above spot, put below)",
    )
    parser.add_argument("--ticker", default="SPX", help="Signal ticker (default SPX)")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Open a paper or live position (default: print signal only)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Allow live Webull orders (sets GEX_TRADER_PAPER=0 for this run)",
    )
    parser.add_argument("--json", action="store_true", help="Print full result JSON")
    parser.add_argument(
        "--any-time",
        action="store_true",
        help="Run outside regular session hours",
    )
    parser.add_argument(
        "--reenter-each-bar",
        action="store_true",
        help="Close any open position before opening toward the new lowest GEX wall",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously (use with --interval)",
    )
    def _default_loop_seconds() -> int:
        raw = os.environ.get("GEX_LOW_GEX_LOOP_SECONDS", "").strip()
        if raw:
            return max(30, int(raw))
        minutes = float(os.environ.get("GEX_LOW_GEX_BAR_MINUTES", os.environ.get("GEX_BACKFILL_INTERVAL_MINUTES", "5")))
        return max(30, int(minutes * 60))

    parser.add_argument(
        "--interval",
        type=int,
        default=_default_loop_seconds(),
        help="Seconds between loop iterations (default: bar minutes × 60, usually 300)",
    )
    args = parser.parse_args()

    if args.live:
        os.environ["GEX_TRADER_PAPER"] = "0"
    if args.reenter_each_bar:
        os.environ["GEX_LOW_GEX_REENTER_EACH_BAR"] = "1"

    def _run_once() -> dict:
        return run_low_gex_trade(
            ticker=args.ticker.upper(),
            execute=args.execute,
            session_check=not args.any_time,
            reenter_each_bar=args.reenter_each_bar,
        )

    def _print_result(result: dict) -> None:
        if args.json:
            print(json.dumps(result, indent=2, default=str))
            return

        ticker = result.get("ticker", args.ticker.upper())
        print(f"\n=== Low-GEX trader · {ticker} ===")
        print(f"broker: {result.get('broker_mode', 'n/a')}")

        if not result.get("ran"):
            print(f"status: skipped — {result.get('reason', 'unknown')}")
            return

        sig = result.get("signal") or {}
        if sig.get("available"):
            rec = sig.get("recommended") or {}
            print(f"spot:   {sig.get('spot', 0):.2f}")
            print(f"wall:   {rec.get('wall_strike', 0):.0f}  (γ {rec.get('gamma_bn', 0):+.3f} Bn)")
            print(f"trade:  {rec.get('option_type', '').upper()} @ {rec.get('strike', 0):.0f}")
            print(f"note:   {rec.get('rationale', '')}")
        else:
            print(f"signal: none — {sig.get('reason', 'unavailable')}")

        closed = result.get("closed_for_rotation") or []
        if closed:
            print(f"rotated: closed {len(closed)} position(s) before entry")

        exits = result.get("exits") or {}
        eod = exits.get("eod_exits") or []
        regular = exits.get("exits") or []
        if eod or regular:
            print(f"exits: {len(regular)} stop/target | {len(eod)} eod flatten")

        action = result.get("action")
        if action == "signal_only":
            print("\n(dry run — pass --execute to open a position)")
        elif action == "opened":
            print(
                f"\nopened: {result.get('option_type', '').upper()} "
                f"{result.get('strike', 0):.2f} x{result.get('qty', 1)} "
                f"@ ${result.get('premium', 0):.2f} ({result.get('broker', '')})"
            )
        elif action:
            print(f"\naction: {action} — {result.get('reason', '')}")

    if args.loop:
        import time

        print(f"Looping every {args.interval}s — Ctrl+C to stop")
        while True:
            _print_result(_run_once())
            time.sleep(max(30, args.interval))
    else:
        _print_result(_run_once())


if __name__ == "__main__":
    main()
