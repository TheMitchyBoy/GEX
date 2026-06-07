"""CLI: Monte Carlo sweep over AI advisor confidence thresholds for ROI optimization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.trading.monte_carlo_confidence import (
    run_confidence_monte_carlo,
    summarize_confidence_monte_carlo,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep min-entry and strong-confidence advisor thresholds on export history.",
    )
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--starting-capital", type=float, default=None)
    parser.add_argument("--min-conf-start", type=float, default=0.35)
    parser.add_argument("--min-conf-stop", type=float, default=0.90)
    parser.add_argument("--min-conf-step", type=float, default=0.05)
    parser.add_argument(
        "--strong-levels",
        default="0.70,0.75,0.80,0.85,0.90",
        help="Comma-separated GEX_TRADER_STRONG_CONFIDENCE values",
    )
    parser.add_argument("--compact", action="store_true", help="Emit summarized JSON only")
    args = parser.parse_args()

    strong_levels = [float(x.strip()) for x in args.strong_levels.split(",") if x.strip()]
    summary = run_confidence_monte_carlo(
        ticker=args.ticker.upper(),
        lookback_days=args.lookback_days,
        max_snapshots=args.max_snapshots,
        starting_capital=args.starting_capital,
        min_conf_start=args.min_conf_start,
        min_conf_stop=args.min_conf_stop,
        min_conf_step=args.min_conf_step,
        strong_levels=strong_levels,
    )
    payload = summarize_confidence_monte_carlo(summary) if args.compact else summary
    print(json.dumps(payload, indent=2, default=str))

    best = summary.get("best_roi") or summary.get("best")
    if best and best.get("total_trades", 0) > 0:
        env = (best.get("config") or {}).get("env") or {}
        print(
            f"\nBest ROI: {best.get('return_pct', 0):.1%} "
            f"({best.get('total_trades')} trades) — "
            f"min_conf={env.get('GEX_TRADER_MIN_ENTRY_CONFIDENCE', '?')}, "
            f"strong_conf={env.get('GEX_TRADER_STRONG_CONFIDENCE', '?')}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
