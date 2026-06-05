# Configuration

Environment variables control data quality, refresh cadence, alerts, and optional features.

## Required

| Variable | Description |
|----------|-------------|
| `UW_API_KEY` | Unusual Whales API key |

## Refresh & scheduler

| Variable | Default | Description |
|----------|---------|-------------|
| `GEX_REFRESH_INTERVAL_MINUTES` | `1` | Background refresh interval for web app |
| `GEX_DISABLE_SCHEDULER` | off | Set `1` to disable APScheduler refresh |
| `GEX_DEFAULT_TICKERS` | `SPX` | Tickers for batch refresh scripts |
| `GEX_UW_CACHE_TTL_SECONDS` | aligned to refresh | Live UW API cache TTL |

## Data quality filters

| Variable | Default | Description |
|----------|---------|-------------|
| `GEX_DATA_FILTERS` | `1` | Master switch (`0` = parse only) |
| `GEX_MIN_OPEN_INTEREST` | `1` | Minimum OI per contract |
| `GEX_MIN_GAMMA` | `0` | Drop non-positive gamma |
| `GEX_MAX_STRIKE_DISTANCE_PCT` | `0.35` | Max strike distance from spot |
| `GEX_MAX_IV` | `6.0` | IV outlier cap |
| `GEX_MAX_BID_ASK_SPREAD_PCT` | `1.0` | Wide/crossed spread filter |
| `GEX_DEDUPE_SYMBOLS` | `1` | Deduplicate symbols |

See [Data Quality](Data-Quality) for filter semantics.

**Stricter SPX example:**

```bash
export GEX_MIN_OPEN_INTEREST=10
export GEX_MAX_STRIKE_DISTANCE_PCT=0.25
export GEX_MAX_BID_ASK_SPREAD_PCT=0.5
```

## Storage & exports

| Variable | Default | Description |
|----------|---------|-------------|
| `GEX_INDEX_DB` | `data/gex_index.db` | SQLite export index path |
| `GEX_EXPORT_DIR` | `data/exports` | Override export directory (scripts) |

## Live flow & alerts

| Variable | Default | Description |
|----------|---------|-------------|
| `GEX_FLOW_FEED` | `data/flow_sample.jsonl` | JSONL flow events for overlay |
| `GEX_ALERT_WEBHOOK_URL` | off | POST target for alert payloads |
| `GEX_ALERT_AUTO_DISPATCH` | off | Auto-send high-severity alerts |
| `GEX_ALERT_DISPATCH_COOLDOWN_MINUTES` | `15` | Min minutes between auto dispatches |
| `GEX_ALERT_AUTO_MIN_SEVERITY` | `high` | `high`, `medium`, or `low` |
| `GEX_ALERT_WALL_SHIFT_PTS` | `20` | Wall migration alert threshold |
| `GEX_ALERT_NEAR_TERM_SPIKE` | `0.12` | Near-term ratio jump threshold |
| `GEX_ALERT_REGIME_FLIP_PROB` | `0.55` | Forecast flip-probability alert |
| `GEX_ALERT_LARGE_DELTA_BN` | `3.0` | Large ΔGEX alert (Bn$) |
| `GEX_ALERT_LARGE_DELTA_RATIO` | `0.25` | Large ΔGEX vs \|total GEX\| |

## Optional AI narrative

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Optional OpenAI polish for `ai_analyst` narrative |

## Example env file

Copy [config/spx.env.example](https://github.com/TheMitchyBoy/GEX/blob/main/config/spx.env.example) and source it before running.
