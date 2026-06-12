#!/usr/bin/env python3
"""Monte Carlo tune for /near wall GEX (±1% window) + wall-shift anti-flicker sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.env_bootstrap import load_env_files

load_env_files()
import os

os.environ.setdefault("GEX_WALL_SIGNAL_FILTERS", "0")

from gex_core.trading.monte_carlo_near_walls import run_near_wall_full_search


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.1%}"


def _print_trial(label: str, row: dict | None) -> None:
    if not row:
        print(f"{label}: none")
        return
    print(f"\n{label}:")
    print(f"  SL {row['stop_loss']:.1%} · TP {row['take_profit']:.1%} · hold {row['max_hold_bars']} bars")
    if row.get("min_gamma_bn") is not None:
        print(
            f"  filters: min|γ|={row.get('min_gamma_bn')} Bn · entry drift={row.get('min_entry_drift_pts')} pts · "
            f"shift min={row.get('wall_shift_min_pts')} pts · shift cooldown={row.get('wall_shift_cooldown_bars')} bars"
        )
    print(
        f"  trades={row['total_trades']} · WR={_fmt_pct(row.get('win_rate'))} · "
        f"PnL=${row.get('total_pnl_usd', 0):,.2f} · ret={_fmt_pct(row.get('return_pct'))} · "
        f"DD={_fmt_pct(row.get('max_drawdown_pct'))}"
    )
    if row.get("by_exit_reason"):
        print(f"  exits: {row['by_exit_reason']}")
    if row.get("blocked_duplicate"):
        print(f"  blocked duplicates: {row['blocked_duplicate']}")
    print(f"  score: {row.get('score', 0):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MC tune near-spot wall GEX")
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--starting-capital", type=float, default=500.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out = run_near_wall_full_search(
        ticker=args.ticker.upper(),
        lookback_days=args.lookback_days,
        starting_capital=args.starting_capital,
    )

    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return

    p1 = out.get("phase1") or {}
    p2 = out.get("phase2") or {}

    print(f"\n=== Near wall GEX MC · {p1.get('ticker', args.ticker)} · window ±{p1.get('window_pct', 0.01)*100:.1f}% ===")
    if p1.get("date_from"):
        print(f"window: {p1['date_from']} -> {p1['date_to']} ({p1.get('snapshots')} snapshots, {args.lookback_days}d lookback)")
    print(f"phase 1 trials: {p1.get('trials_run', 0)}")

    _print_trial("Phase 1 best (SL/TP/hold)", p1.get("best_profitable") or p1.get("best"))

    if p1.get("top"):
        print("\nPhase 1 top 8:")
        for row in p1["top"][:8]:
            print(
                f"  SL {row['stop_loss']:4.0%} TP {row['take_profit']:4.0%} hold {row['max_hold_bars']:2d} · "
                f"ret={_fmt_pct(row.get('return_pct')):>7} pnl=${row.get('total_pnl_usd', 0):7.2f} "
                f"trades={row['total_trades']:3d} wall_shift={row.get('by_exit_reason', {}).get('wall_shift', 0)}"
            )

    if p2:
        print(f"\nphase 2 trials: {p2.get('trials_run', 0)} (wall-shift / filter sweep)")
        _print_trial("Phase 2 baseline (no extra filters)", p2.get("baseline_no_filters"))
        _print_trial("Phase 2 best tuned", p2.get("best_profitable") or p2.get("best"))

        if p2.get("top"):
            print("\nPhase 2 top 8:")
            for row in p2["top"][:8]:
                print(
                    f"  |γ|>={row.get('min_gamma_bn')} drift>={row.get('min_entry_drift_pts')} "
                    f"shift>={row.get('wall_shift_min_pts')} cd={row.get('wall_shift_cooldown_bars')} · "
                    f"ret={_fmt_pct(row.get('return_pct')):>7} pnl=${row.get('total_pnl_usd', 0):7.2f} "
                    f"trades={row['total_trades']:3d} wall_shift={row.get('by_exit_reason', {}).get('wall_shift', 0)}"
                )


if __name__ == "__main__":
    main()
