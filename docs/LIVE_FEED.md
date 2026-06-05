# Live option flow feed

The `live/` package ingests JSON-lines flow events and estimates per-strike GEX deltas. The Flask dashboard blends this feed into forecasts when `GEX_FLOW_FEED` points at a readable file (default: `data/flow_sample.jsonl`).

## Event format

```json
{
  "option": "SPX260620C04800000",
  "gamma": 0.00012,
  "quantity": 50,
  "side": "buy",
  "spot": 4800.0
}
```

## Running the sample ingest loop

```bash
python live/ingest.py --feed data/flow_sample.jsonl --spot 4800
```

## Production adapter

1. Tail a broker or vendor JSONL file into `data/live_flow.jsonl`.
2. Set `export GEX_FLOW_FEED=data/live_flow.jsonl`.
3. Optionally enable auto alert dispatch: `GEX_ALERT_AUTO_DISPATCH=1` and `GEX_ALERT_WEBHOOK_URL`.

For websocket sources, implement the same loop as `live/ingest.py`: parse events, append JSON lines, and let the dashboard call `load_flow_predictions()` on each request.
