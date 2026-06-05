# Operations

## Scheduled exports

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `daily_exports.yml` | Daily 06:00 UTC | Full SPX snapshot commit |
| `intraday_exports.yml` | Weekdays, every 30m (14–21 UTC) | Intraday history density |

Both require `UW_API_KEY` in repository secrets.

## CI

`ci.yml` on push/PR:

- `pytest` unit tests  
- `backtest_gex_prediction.py`  
- Sign-accuracy gate when walk-forward `n ≥ 4` (floor `GEX_BACKTEST_MIN_ACCURACY`, default 0.30)  

## Health monitoring

```bash
curl -sf http://localhost:8080/health || echo "unhealthy"
```

Watch `export_age_minutes` vs `GEX_REFRESH_INTERVAL_MINUTES` (stale if &gt; 3× interval).

## Export retention

Compact old strike CSVs while keeping summaries and cumulative series:

```bash
python scripts/gex_compact_exports.py --ticker SPX --keep-full-days 14
python scripts/gex_compact_exports.py --dry-run  # preview
```

## SQLite index

`data/gex_index.db` accelerates timestamp lookups. Rebuilt on each export via `gex_core.storage.upsert_snapshot`. To reindex:

```python
from gex_core.storage import sync_ticker_exports
sync_ticker_exports("SPX")
```

## Docker

```bash
docker compose up web
```

Mounts `./data` for persistent exports. Set `UW_API_KEY` in environment.

## Wiki publishing

Changes under `wiki/` on `main` trigger `.github/workflows/publish-wiki.yml`, which syncs to the GitHub Wiki.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Empty dashboard | `data/exports/` has SPX files; run `gex_refresh.py --force` |
| No live UW overlay | `UW_API_KEY` set; `GEX_SHOW_UW_LIVE` not disabled |
| Forecast missing | Need ≥4 snapshots; see `MIN_KNN_SNAPSHOTS` |
| Model overlay inactive | Manifest `n_train` &lt; 8; train or wait for history |
| Alerts not sending | `GEX_ALERT_WEBHOOK_URL`; cooldown / dedupe for auto dispatch |
