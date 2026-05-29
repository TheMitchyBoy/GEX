"""Live ingest CLI for option flow feeds.

This simple tool supports reading JSON-lines feeds (local or streamed) and
passing each event to the GEXAggregator. It's a pluggable adapter; production
adapters (websocket, broker APIs) should implement the same read loop.

Run example (reads packaged sample):
    python live/ingest.py --feed data/flow_sample.jsonl --spot 4800
"""
import argparse
import json
from pathlib import Path
import time

try:
    from live.aggregator import EnhancedGEXAggregator
except ImportError:
    from aggregator import EnhancedGEXAggregator


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def main(feed: str, spot: float = None, top_n: int = 5, delay: float = 0.5):
    path = Path(feed)
    if not path.exists():
        raise SystemExit(f"Feed not found: {feed}")

    # Use enhanced aggregator for better signals
    agg = EnhancedGEXAggregator(spot=spot)
    for evt in read_jsonl(path):
        # inject spot if provided
        if spot is not None and "spot" not in evt:
            evt["spot"] = spot
        try:
            info = agg.ingest_event(evt)
            strike = info["strike"]
            delta = info["delta_gex"]
            sig = info["signal"]
            print(f"Event -> strike={strike}, delta_gex={delta:,.2f}, signal={sig['score']:.3f} ({sig['direction']})")
        except Exception as e:
            print("Skipping event:", e)
        movers = agg.top_signals(top_n=top_n)
        print("Top signals:")
        for s, g in movers:
            print(f"  strike {s}: score={g['score']:.3f}, dir={g['direction']}, recent_gex={g['recent_gex']:,.2f}")
        print("---")
        time.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed", required=True, help="Path to JSONL feed file")
    parser.add_argument("--spot", type=float, default=None, help="Spot price to use for computations")
    parser.add_argument("--top-n", type=int, default=5, help="Top N movers to display")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between events (s)")
    args = parser.parse_args()
    main(args.feed, spot=args.spot, top_n=args.top_n, delay=args.delay)
