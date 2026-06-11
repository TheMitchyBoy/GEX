#!/usr/bin/env python3
"""Backtest each suggested trader improvement in isolation over a lookback window."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.exports import EXPORT_DIR
from gex_core.history import _build_history_impl
from gex_core.trading.backtest import backtest_auto_trader

# Production defaults (matches config.py fallbacks on main).
PRODUCTION_BASELINE: dict[str, str] = {
    "GEX_TRADER_STRICT_FILTERS": "1",
    "GEX_TRADER_REQUIRE_MOMENTUM": "1",
    "GEX_TRADER_REQUIRE_FLIP_SIDE": "1",
    "GEX_TRADER_REQUIRE_FLOW_ALIGN": "0",
    "GEX_TRADER_MAX_GAMMA_ONLY": "1",
    "GEX_TRADER_MAGNET_ANCHORED_STRIKES": "0",
    "GEX_TRADER_FIX_MAGNET_EXIT_SCALE": "1",
    "GEX_TRADER_MIN_MAGNET_PROGRESS": "0",
    "GEX_TRADER_MIN_MAGNET_DISTANCE_PCT": "0",
    "GEX_TRADER_STOP_LOSS_PCT": "0.03",
    "GEX_TRADER_TAKE_PROFIT_PCT": "0.22",
    "GEX_TRADER_MAX_HOLD_MINUTES": "30",
    "GEX_TRADER_MAGNET_TOUCH_EXIT": "0",
    "GEX_TRADER_DYNAMIC_TP": "0",
    "GEX_TRADER_DYNAMIC_TIME_STOP": "0",
    "GEX_TRADER_EQUITY_FROM_MARK": "0",
    "GEX_TRADER_REGIME_STRICT": "0",
    "GEX_TRADER_MAGNET_PARTIAL_EXIT": "0",
    "GEX_TRADER_MIN_ENTRY_CONFIDENCE": "0",
    "GEX_TRADER_STRONG_CONFIDENCE": "0.80",
    "GEX_TRADER_MAX_OPEN": "2",
    "GEX_TRADER_MAX_ENTRIES_PER_CYCLE": "1",
    "GEX_TRADER_MULTI_STRIKE": "2",
    "GEX_TRADER_ENTRY_TIME_FILTER": "1",
    "GEX_TRADER_ENTRY_AFTER_OPEN_MIN": "15",
    "GEX_TRADER_ENTRY_BEFORE_CLOSE_MIN": "30",
    "GEX_TRADER_TIME_STOP_MIN_PROGRESS": "0.35",
    "GEX_TRADER_TAKE_PROFIT_PCT": "0.35",
    "GEX_TRADER_MAGNET_PARTIAL_PROGRESS": "0.80",
}

SCENARIOS: list[tuple[str, dict[str, str]]] = [
    ("★ production baseline", {}),
    # --- Original sweep ---
    ("1. flow align ON (legacy strict)", {"GEX_TRADER_REQUIRE_FLOW_ALIGN": "1"}),
    ("2. entry min magnet progress 15%", {"GEX_TRADER_MIN_MAGNET_PROGRESS": "0.15"}),
    ("3. min magnet distance 0.3%", {"GEX_TRADER_MIN_MAGNET_DISTANCE_PCT": "0.003"}),
    ("4. confidence floor 0.85", {"GEX_TRADER_MIN_ENTRY_CONFIDENCE": "0.85", "GEX_TRADER_STRONG_CONFIDENCE": "0.70"}),
    ("5. max_open=1", {"GEX_TRADER_MAX_OPEN": "1", "GEX_TRADER_MAX_ENTRIES_PER_CYCLE": "1"}),
    ("6. regime strict (block short γ)", {"GEX_TRADER_REGIME_STRICT": "1"}),
    ("7. wider entry window (30/45 min)", {"GEX_TRADER_ENTRY_AFTER_OPEN_MIN": "30", "GEX_TRADER_ENTRY_BEFORE_CLOSE_MIN": "45"}),
    # --- Exit / time-stop tuning ---
    ("8. time_stop min progress 35%", {"GEX_TRADER_TIME_STOP_MIN_PROGRESS": "0.35"}),
    ("9. time_stop bars 10", {"GEX_TRADER_TIME_STOP_BARS": "10"}),
    ("10. time_stop bars 4 (tighter)", {"GEX_TRADER_TIME_STOP_BARS": "4"}),
    ("11. magnet partial exit @80%", {"GEX_TRADER_MAGNET_PARTIAL_EXIT": "1"}),
    ("12. magnet partial exit @75%", {"GEX_TRADER_MAGNET_PARTIAL_EXIT": "1", "GEX_TRADER_MAGNET_PARTIAL_PROGRESS": "0.75"}),
    ("13. take_profit 15%", {"GEX_TRADER_TAKE_PROFIT_PCT": "0.15"}),
    ("14. take_profit 20%", {"GEX_TRADER_TAKE_PROFIT_PCT": "0.20"}),
    ("15. partial TP 12% (non-hold setups)", {"GEX_TRADER_PARTIAL_TP_PCT": "0.12"}),
    # --- Entry / exposure ---
    ("16. multi_strike=1", {"GEX_TRADER_MULTI_STRIKE": "1"}),
    ("17. flow OFF + multi_strike=1", {"GEX_TRADER_REQUIRE_FLOW_ALIGN": "0", "GEX_TRADER_MULTI_STRIKE": "1"}),
    # --- Combos (positive levers only) ---
    (
        "18. flow OFF + time_stop progress 35%",
        {"GEX_TRADER_REQUIRE_FLOW_ALIGN": "0", "GEX_TRADER_TIME_STOP_MIN_PROGRESS": "0.35"},
    ),
    (
        "19. flow OFF + magnet partial @75%",
        {"GEX_TRADER_REQUIRE_FLOW_ALIGN": "0", "GEX_TRADER_MAGNET_PARTIAL_EXIT": "1", "GEX_TRADER_MAGNET_PARTIAL_PROGRESS": "0.75"},
    ),
    (
        "20. flow OFF + multi_strike=1 + magnet partial @75%",
        {
            "GEX_TRADER_REQUIRE_FLOW_ALIGN": "0",
            "GEX_TRADER_MULTI_STRIKE": "1",
            "GEX_TRADER_MAGNET_PARTIAL_EXIT": "1",
            "GEX_TRADER_MAGNET_PARTIAL_PROGRESS": "0.75",
        },
    ),
    (
        "21. flow OFF + time_stop 35% + magnet partial @75%",
        {
            "GEX_TRADER_REQUIRE_FLOW_ALIGN": "0",
            "GEX_TRADER_TIME_STOP_MIN_PROGRESS": "0.35",
            "GEX_TRADER_MAGNET_PARTIAL_EXIT": "1",
            "GEX_TRADER_MAGNET_PARTIAL_PROGRESS": "0.75",
        },
    ),
]

_TRACKED_KEYS = sorted(
    {k for env in [PRODUCTION_BASELINE, *[o for _, o in SCENARIOS]] for k in env} | set(PRODUCTION_BASELINE)
)


@contextmanager
def trader_env(overrides: dict[str, str]):
    saved = {k: os.environ.get(k) for k in _TRACKED_KEYS}
    try:
        for k in _TRACKED_KEYS:
            os.environ.pop(k, None)
        for k, v in PRODUCTION_BASELINE.items():
            os.environ[k] = v
        for k, v in overrides.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_case(name: str, overrides: dict[str, str], history: list[dict], capital: float) -> dict:
    with trader_env(overrides):
        result = backtest_auto_trader("SPX", history=history, starting_capital=capital)
    acct = result.get("account") or {}
    by_exit = result.get("by_exit_reason") or {}
    return {
        "name": name,
        "trades": int(result.get("total_trades") or 0),
        "wins": int(result.get("wins") or 0),
        "win_rate": float(result.get("win_rate") or 0),
        "pnl_usd": float(result.get("total_pnl_usd") or 0),
        "return_pct": float(acct.get("return_pct") or 0),
        "max_dd": float(acct.get("max_drawdown_pct") or 0),
        "skipped_filters": int(result.get("skipped_filters") or 0),
        "magnet_touch": int(by_exit.get("magnet_touch") or 0),
        "magnet_partial": int(by_exit.get("magnet_partial") or 0),
        "time_stop": int(by_exit.get("time_stop") or 0),
        "stop_loss": int(by_exit.get("stop_loss") or 0),
        "take_profit": int(by_exit.get("take_profit") or 0),
        "eod_flatten": int(by_exit.get("eod_flatten") or 0),
        "trailing_stop": int(by_exit.get("trailing_stop") or 0),
        "date_from": result.get("date_from"),
        "date_to": result.get("date_to"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep suggested trader improvements via backtest")
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--starting-capital", type=float, default=500)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", help="Write JSON results to this path")
    args = parser.parse_args()

    history = _build_history_impl(
        args.ticker.upper(),
        EXPORT_DIR,
        lookback_days=args.lookback_days,
        max_snapshots=500,
        dedupe_identical_strikes=False,
    )
    if len(history) < 2:
        print(json.dumps({"error": "insufficient history", "snapshots": len(history)}))
        return 1

    rows = [_run_case(name, overrides, history, args.starting_capital) for name, overrides in SCENARIOS]
    baseline_pnl = rows[0]["pnl_usd"]
    for row in rows:
        row["pnl_vs_baseline"] = round(row["pnl_usd"] - baseline_pnl, 2)

    payload = {"snapshots": len(history), "lookback_days": args.lookback_days, "results": rows}

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print("=" * 115)
    print(
        f"IMPROVEMENT SWEEP — {args.ticker} {args.lookback_days}d | {len(history)} snapshots | "
        f"${args.starting_capital:.0f} start"
    )
    print(f"Window: {rows[0]['date_from']} -> {rows[0]['date_to']}")
    print("=" * 115)
    print(
        f"{'Scenario':<44} {'Trd':>4} {'Win%':>6} {'PnL':>9} {'ΔPnL':>8} {'ROI':>6} "
        f"{'MagT':>5} {'MagP':>5} {'TStop':>5} {'TP':>4} {'EOD':>4}"
    )
    print("-" * 115)
    for row in rows:
        print(
            f"{row['name']:<44} {row['trades']:>4} {row['win_rate']*100:>5.1f}% "
            f"${row['pnl_usd']:>8.2f} {row['pnl_vs_baseline']:>+8.2f} {row['return_pct']*100:>5.1f}% "
            f"{row['magnet_touch']:>5} {row['magnet_partial']:>5} {row['time_stop']:>5} "
            f"{row['take_profit']:>4} {row['eod_flatten']:>4}"
        )

    ranked = sorted(rows[1:], key=lambda r: r["pnl_usd"], reverse=True)
    print()
    print("Top 5 vs production baseline:")
    for row in ranked[:5]:
        print(
            f"  {row['name']}: ${row['pnl_usd']:.2f} (Δ{row['pnl_vs_baseline']:+.2f}), "
            f"{row['trades']} trades, {row['win_rate']*100:.1f}% win"
        )
    print()
    worst = min(rows[1:], key=lambda r: r["pnl_usd"])
    print(f"Worst: {worst['name']} — ${worst['pnl_usd']:.2f} (Δ{worst['pnl_vs_baseline']:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
