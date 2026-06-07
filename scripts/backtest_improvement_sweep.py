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

# Baseline matches current production defaults (all new toggles off).
BASELINE_ENV: dict[str, str] = {
    "GEX_TRADER_STRICT_FILTERS": "1",
    "GEX_TRADER_REQUIRE_MOMENTUM": "1",
    "GEX_TRADER_REQUIRE_FLIP_SIDE": "1",
    "GEX_TRADER_REQUIRE_FLOW_ALIGN": "0",
    "GEX_TRADER_MAX_GAMMA_ONLY": "1",
    "GEX_TRADER_MAGNET_ANCHORED_STRIKES": "0",
    "GEX_TRADER_FIX_MAGNET_EXIT_SCALE": "0",
    "GEX_TRADER_MIN_MAGNET_PROGRESS": "0",
    "GEX_TRADER_MIN_MAGNET_DISTANCE_PCT": "0",
    "GEX_TRADER_DYNAMIC_TIME_STOP": "0",
    "GEX_TRADER_EQUITY_FROM_MARK": "0",
    "GEX_TRADER_REGIME_STRICT": "0",
    "GEX_TRADER_MAGNET_PARTIAL_EXIT": "0",
    "GEX_TRADER_MIN_ENTRY_CONFIDENCE": "0",
    "GEX_TRADER_STRONG_CONFIDENCE": "0.80",
    "GEX_TRADER_MAX_OPEN": "2",
    "GEX_TRADER_ENTRY_TIME_FILTER": "1",
    "GEX_TRADER_ENTRY_AFTER_OPEN_MIN": "15",
    "GEX_TRADER_ENTRY_BEFORE_CLOSE_MIN": "30",
}

SCENARIOS: list[tuple[str, dict[str, str]]] = [
    ("baseline (current)", {}),
    ("1. fix magnet exit scale (SPX→SPY)", {"GEX_TRADER_FIX_MAGNET_EXIT_SCALE": "1"}),
    ("2. entry min magnet progress 15%", {"GEX_TRADER_MIN_MAGNET_PROGRESS": "0.15"}),
    ("3. min magnet distance 0.3%", {"GEX_TRADER_MIN_MAGNET_DISTANCE_PCT": "0.003"}),
    ("4. dynamic time_stop by distance", {"GEX_TRADER_DYNAMIC_TIME_STOP": "1"}),
    ("5. confidence floor 0.85", {"GEX_TRADER_MIN_ENTRY_CONFIDENCE": "0.85", "GEX_TRADER_STRONG_CONFIDENCE": "0.70"}),
    ("6. equity from mark (fix ROI)", {"GEX_TRADER_EQUITY_FROM_MARK": "1"}),
    ("7. max_open=1 (dedupe exposure)", {"GEX_TRADER_MAX_OPEN": "1", "GEX_TRADER_MAX_ENTRIES_PER_CYCLE": "1"}),
    ("8. regime strict (block short γ)", {"GEX_TRADER_REGIME_STRICT": "1"}),
    ("9. magnet partial exit @80%", {"GEX_TRADER_MAGNET_PARTIAL_EXIT": "1"}),
    (
        "10. wider entry window (30/45 min)",
        {"GEX_TRADER_ENTRY_AFTER_OPEN_MIN": "30", "GEX_TRADER_ENTRY_BEFORE_CLOSE_MIN": "45"},
    ),
    (
        "★ best combo (1+2+5+7)",
        {
            "GEX_TRADER_FIX_MAGNET_EXIT_SCALE": "1",
            "GEX_TRADER_MIN_MAGNET_PROGRESS": "0.15",
            "GEX_TRADER_MIN_ENTRY_CONFIDENCE": "0.85",
            "GEX_TRADER_STRONG_CONFIDENCE": "0.70",
            "GEX_TRADER_MAX_OPEN": "1",
            "GEX_TRADER_MAX_ENTRIES_PER_CYCLE": "1",
        },
    ),
]

_TRACKED_KEYS = sorted({k for env in [BASELINE_ENV, *[o for _, o in SCENARIOS]] for k in env} | set(BASELINE_ENV))


@contextmanager
def trader_env(overrides: dict[str, str]):
    saved = {k: os.environ.get(k) for k in _TRACKED_KEYS}
    try:
        for k in _TRACKED_KEYS:
            os.environ.pop(k, None)
        for k, v in BASELINE_ENV.items():
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
        "date_from": result.get("date_from"),
        "date_to": result.get("date_to"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep suggested trader improvements via backtest")
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--starting-capital", type=float, default=500)
    parser.add_argument("--json", action="store_true")
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

    if args.json:
        print(json.dumps({"snapshots": len(history), "results": rows}, indent=2))
        return 0

    print("=" * 100)
    print(f"IMPROVEMENT SWEEP — {args.ticker} {args.lookback_days}d | {len(history)} snapshots | ${args.starting_capital:.0f} start")
    print(f"Window: {rows[0]['date_from']} -> {rows[0]['date_to']}")
    print("=" * 100)
    print(
        f"{'Scenario':<42} {'Trades':>6} {'Win%':>6} {'PnL':>9} {'ΔPnL':>8} "
        f"{'MagTouch':>8} {'TimeStop':>8} {'SkipFilt':>8}"
    )
    print("-" * 100)
    for row in rows:
        print(
            f"{row['name']:<42} {row['trades']:>6} {row['win_rate']*100:>5.1f}% "
            f"${row['pnl_usd']:>8.2f} {row['pnl_vs_baseline']:>+8.2f} "
            f"{row['magnet_touch']:>8} {row['time_stop']:>8} {row['skipped_filters']:>8}"
        )

    best = max(rows[1:], key=lambda r: r["pnl_usd"])
    print()
    print(f"Best single change vs baseline: {best['name']} (PnL ${best['pnl_usd']:.2f}, Δ{best['pnl_vs_baseline']:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
