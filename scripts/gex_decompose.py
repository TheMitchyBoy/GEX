"""CLI: decompose GEX changes between UW export snapshots.

Usage:
    python scripts/gex_decompose.py --ticker SPX
    python scripts/gex_decompose.py --ticker SPX --flow data/flow_sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex_core.decompose import decompose_from_snapshots
from gex_core.exports import find_exports_for_ticker


def load_flow_events(path: Path) -> list[dict]:
    events = []
    if not path.exists():
        return events
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def print_decomposition(d, title: str = "GEX Decomposition"):
    print(f"\n=== {title} ===")
    print(f"  Baseline total GEX:     {d.baseline_total_bn:+.4f} Bn$ / %")
    print(f"  Spot shift component:   {d.spot_shift_total_bn:+.4f} Bn$ / %  (spot {d.spot_pct:+.2%})")
    print(f"  Time decay component:   {d.time_shift_total_bn:+.4f} Bn$ / %  ({d.hours_elapsed:.1f} h)")
    print(f"  Vol shift component:    {d.vol_shift_total_bn:+.4f} Bn$ / %  (vol {d.vol_pct:+.2%})")
    print(f"  Flow component:         {d.flow_total_bn:+.4f} Bn$ / %")
    print(f"  Predicted total GEX:    {d.predicted_total_bn:+.4f} Bn$ / %")
    if d.actual_next_total_bn is not None:
        residual = d.actual_next_total_bn - d.predicted_total_bn
        print(f"  Actual next total GEX:  {d.actual_next_total_bn:+.4f} Bn$ / %")
        print(f"  Residual:               {residual:+.4f} Bn$ / %")


def main():
    parser = argparse.ArgumentParser(description="Decompose GEX changes between snapshots")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--flow", type=Path, default=None, help="JSONL flow feed path")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    flow_events = load_flow_events(args.flow) if args.flow else []

    exports = find_exports_for_ticker(ticker)
    timestamps = sorted(ts for ts, kinds in exports.items() if "gex_by_strike" in kinds)
    if len(timestamps) < 2:
        print("Need at least 2 snapshots with gex_by_strike exports.")
        sys.exit(1)

    ts_prev, ts_next = timestamps[-2], timestamps[-1]
    d = decompose_from_snapshots(
        exports[ts_prev]["gex_by_strike"],
        exports[ts_next]["gex_by_strike"],
        ticker,
        ts_prev,
        ts_next,
        flow_events=flow_events,
    )
    print_decomposition(
        d,
        title=f"{ticker} snapshot comparison ({ts_prev} → {ts_next})",
    )


if __name__ == "__main__":
    main()
