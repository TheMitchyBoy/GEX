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
    args = parser.parse_args()

    if args.live:
        os.environ["GEX_TRADER_PAPER"] = "0"

    result = run_low_gex_trade(
        ticker=args.ticker.upper(),
        execute=args.execute,
        session_check=not args.any_time,
    )

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


if __name__ == "__main__":
    main()
