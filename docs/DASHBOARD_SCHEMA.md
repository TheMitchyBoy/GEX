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

Per-strike gamma profile for each snapshot. Rows are filtered to **±12% of spot** by default (see `GEX_STORE_STRIKE_DISTANCE_PCT`); far OTM strikes like 200 are UW chain noise and are excluded from storage. Full chain remains in `greek_exposure_json` when fetched.

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

| Column | Type | Description |
|--------|------|-------------|
| `quality_score` | DOUBLE | 0–1 composite data quality score |
| `flip_confidence` | TEXT | Gamma flip estimate confidence (`high` / `medium` / `low` / `none`) |
| `regime_consistent` | BOOLEAN | Total GEX regime matches cumulative slope at spot |
| `spot_source` | TEXT | Which UW spot candidate was chosen |
| `spot_disagreement_pct` | DOUBLE | Max spread across spot sources / spot |
| `strike_profile_confidence` | TEXT | `high` / `medium` / `low` based on strike profile source |
| `data_lag_sec` | DOUBLE | Seconds between UW observation time and write |
| `uw_rate_limit_json` | JSONB | Last UW rate-limit response headers |

## snapshot_diagnostics

Write pipeline status (`ok`, `ok_with_warnings`, `skipped_duplicate`, `rejected`) plus timing metrics.

| Column | Type | Description |
|--------|------|-------------|
| `quality_score` | DOUBLE | Same score as `snapshot_features` |
| `data_lag_sec` | DOUBLE | UW data staleness at write time |
| `uw_rate_limit_json` | JSONB | UW API rate-limit metadata |

## daily_quality_stats

Per-ticker, per-day rollup of snapshot write outcomes and rolling averages (`quality_score_avg`, `data_lag_sec_avg`, `uw_fetch_ms_avg`, status counts).

## prediction_accuracy_daily

Per-ticker, per-day rollup of resolved LLM prediction outcomes (`sign_hit_rate`, `bias_hit_rate`, `regime_hit_rate`).

## training_snapshots (view)

Filtered snapshots suitable for model training: `quality_score >= 0.8`, non-low strike confidence, not `eod_scaled`, diagnostic status `ok` or `ok_with_warnings`.

```sql
SELECT ticker, ts, quality_score, flip_confidence, regime_consistent
FROM training_snapshots
WHERE ticker = 'SPX'
ORDER BY ts DESC
LIMIT 20;
```

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

### One-time CSV → Postgres import

If strike CSVs exist on disk (e.g. after `GEX_EXPORT_CSV=1` or a local backfill) but external dashboards need full Postgres payloads (`snapshot_strikes`, `summary_json`, features), run:

```bash
python3 scripts/import_exports_to_postgres.py --tickers SPX
```

Set `GEX_IMPORT_EXPORTS_ON_START=1` on the processor service to import any missing timestamps on boot.

### Catch-up sync (Postgres)

When Postgres already has history but is behind (e.g. latest snapshot is 2026-04-30 while today is June), run:

```bash
python3 scripts/sync_postgres_snapshots.py --tickers SPX
```

This imports any on-disk CSV exports missing from Postgres, then pulls UW intraday history for all trading days after the latest stored `market_date`. Equivalent manual backfill:

```bash
python3 scripts/backfill_postgres_history.py --tickers SPX --catch-up
python3 scripts/import_exports_to_postgres.py --tickers SPX
```

On Railway, processor boot runs the same catch-up logic when the latest snapshot is before today (not only when total count is below 30).

### UW API history backfill (Postgres)

Pull up to 90 days of UW intraday + daily history directly into Postgres (no CSV intermediate):

```bash
python3 scripts/backfill_postgres_history.py --tickers SPX
```

Options:

- `--intraday-days 90` / `--daily-days 90` — lookback window (weekdays)
- `--interval-minutes 10` — downsample UW 1-minute rows (default 10)
- `--if-sparse` — skip when enough snapshots already exist
- `--force` — overwrite existing timestamps

On Railway, bootstrap runs automatically on processor boot when snapshot count is below `GEX_BACKFILL_MIN_SNAPSHOTS` (default **30**). Check progress:

```bash
tail -f /app/data/bootstrap_postgres.log
curl -s https://<your-processor>/health/live | jq .snapshot_count
```

Defaults on processor start:

- `GEX_STARTUP_BACKFILL=1` — import CSVs and catch up UW history when sparse or stale
- `GEX_IMPORT_EXPORTS_ON_START=1` — import local CSV exports when present on disk
- `GEX_BACKFILL_MIN_SNAPSHOTS=30` — threshold before backfill is skipped

Or run manually as a one-off job:

Required env:

- `DATABASE_URL` — Railway PostgreSQL
- `UW_API_KEY` — Unusual Whales API
- `GEX_EXPORT_CSV=0` — default; Postgres only

Optional:

- `GEX_REFRESH_INTERVAL_MINUTES=10`
- `GEX_STARTUP_BACKFILL=1` — first-deploy history pull
- `GEX_DEFAULT_TICKERS=SPX`
- `GEX_HARD_REJECT_TOTAL_GEX_MISMATCH=1` — reject session snapshots when total GEX ≠ strike sum
- `GEX_QUALITY_ALERTS=0` — set `1` to webhook on quality anomalies
- `GEX_MAX_DATA_LAG_SEC=1200` — staleness warning threshold
- `GEX_SPOT_DISAGREEMENT_TOLERANCE_PCT=0.005` — spot cross-check tolerance
- `GEX_MIN_STRIKE_GEX_BN=1e-6` — drop dust strikes before Postgres write

Health: `GET /health/live` returns JSON with `latest_ts` and `status`.
