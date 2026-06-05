# Live Flow

The `live/` package estimates **per-strike GEX deltas** from a stream of option trade events (JSON Lines).

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

| Field | Required | Notes |
|-------|----------|-------|
| `option` | yes | OCC-style symbol |
| `gamma` | yes | Gamma per contract |
| `quantity` | yes | Contract count |
| `side` | no | `buy` / `sell` (default `buy`) |
| `spot` | no | Uses aggregator spot if omitted |

## Sample ingest

```bash
python live/ingest.py --feed data/flow_sample.jsonl --spot 4800
```

## Dashboard integration

1. Point `GEX_FLOW_FEED` at your JSONL file (default: `data/flow_sample.jsonl`).  
2. Open `/ticker/SPX` — flow overlay card and forecast blend appear when events exist.  
3. `load_flow_predictions` + `apply_flow_to_prediction` adjust ΔGEX and strike profile.  

## Production setup

1. Tail broker/vendor events into e.g. `data/live_flow.jsonl`.  
2. `export GEX_FLOW_FEED=data/live_flow.jsonl`  
3. Optional alerts: `GEX_ALERT_AUTO_DISPATCH=1` + `GEX_ALERT_WEBHOOK_URL`  

For **websocket** feeds, append the same JSON schema in a loop (see `live/ingest.py`).

## Future work

A native Unusual Whales websocket adapter is on the [Roadmap](Roadmap).
