# GEX PostgreSQL schema — processor output for external dashboards

Run idempotently in Railway Postgres or via `python3 scripts/init_postgres_schema.py`.

## snapshots

One row per GEX snapshot (intraday slice or EOD).

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | TEXT | e.g. `SPX` |
| `ts` | TEXT | `YYYY-MM-DD_HHMMSS` (UTC export key) |
| `market_date` | TEXT | Trading day `YYYY-MM-DD` |
| `spot` | DOUBLE | Spot at snapshot |
| `total_gex` | DOUBLE | Net GEX (Bn$/1% move) |
| `regime` | TEXT | e.g. `LONG gamma` / `SHORT gamma` |
| `summary_json` | JSONB | Full summary payload (walls, flip, flow, macro) |
| `expiration_json` | JSONB | `{expiration: gex_bn_per_pct}` |
| `surface_json` | JSONB | Surface rows (array of objects) |
| `greek_exposure_json` | JSONB | Greek exposure rows |
| `indexed_at` | TEXT | ISO UTC when row was written |
| `snapshot_at` | TIMESTAMPTZ | Parsed UTC snapshot time |
| `prior_ts` | TEXT | Previous snapshot key |

**Primary key:** `(ticker, ts)`

### Example queries

```sql
-- Latest snapshot
SELECT ticker, ts, spot, total_gex, regime, summary_json
FROM snapshots
WHERE ticker = 'SPX'
ORDER BY ts DESC
LIMIT 1;

-- Timeline for a day
SELECT ts, spot, total_gex, regime
FROM snapshots
WHERE ticker = 'SPX' AND market_date = '2026-06-15'
ORDER BY ts;

-- Extract gamma flip from summary
SELECT ts, summary_json->>'gamma_flip' AS gamma_flip
FROM snapshots
WHERE ticker = 'SPX'
ORDER BY ts DESC
LIMIT 24;
```

## snapshot_strikes

Per-strike gamma profile for each snapshot.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | TEXT | |
| `ts` | TEXT | Matches `snapshots.ts` |
| `strike` | DOUBLE | Strike price |
| `gex_bn_per_pct` | DOUBLE | GEX at strike |
| `cumulative_gex_bn_per_pct` | DOUBLE | Cumulative GEX |

**Primary key:** `(ticker, ts, strike)`

## snapshot_strikes_atm

ATM subset (default ±3% of spot). Same columns as `snapshot_strikes`.

## snapshot_features

Precomputed ML features: `gamma_flip`, `call_wall`, `put_wall`, `delta_gex`, `delta_spot`, `spot_return`, `surface_vector`, `strike_profile_hash`, etc.

## snapshot_diagnostics

Write pipeline status (`ok`, `skipped_duplicate`, `rejected`) plus timing metrics.

## processor_state

Incremental backfill cursor (`backfill_last_date:SPX`).

Subscribe to new snapshots: `LISTEN gex_snapshot;`

Materialized view: `latest_snapshot` (one row per ticker).

### Example queries

```sql
-- Strike profile for one snapshot
SELECT strike, gex_bn_per_pct, cumulative_gex_bn_per_pct
FROM snapshot_strikes
WHERE ticker = 'SPX' AND ts = '2026-06-15_225554'
ORDER BY strike;

-- Join summary + strikes
SELECT s.ts, s.spot, s.regime, st.strike, st.gex_bn_per_pct
FROM snapshots s
JOIN snapshot_strikes st ON st.ticker = s.ticker AND st.ts = s.ts
WHERE s.ticker = 'SPX'
ORDER BY s.ts DESC, st.strike
LIMIT 100;
```

## Other tables (optional for dashboards)

| Table | Purpose |
|-------|---------|
| `trades` | Paper/live trade journal |
| `decisions` | Trader decision log |
| `trader_state` | Key/value trader flags |
| `llm_predictions` | LLM forecast log |
| `daily_insights` | Daily lessons and strategies |

## Processor service

Deploy this repo with:

```bash
bash scripts/start_processor.sh
```

Required env:

- `DATABASE_URL` — Railway PostgreSQL
- `UW_API_KEY` — Unusual Whales API
- `GEX_EXPORT_CSV=0` — default; Postgres only

Optional:

- `GEX_REFRESH_INTERVAL_MINUTES=10`
- `GEX_STARTUP_BACKFILL=1` — first-deploy history pull
- `GEX_DEFAULT_TICKERS=SPX`

Health: `GET /health/live` returns JSON with `latest_ts` and `status`.
