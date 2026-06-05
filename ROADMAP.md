# GEX roadmap

Implemented in this release:

- SQLite export index (`data/gex_index.db`) for fast timestamp lookups
- Export metadata (`export_schema_version`, `filter_config_hash`) in summary JSON
- History LRU cache + walk-forward backtest gates in CI
- `/health` operational endpoint
- Configurable alert thresholds and optional auto webhook dispatch
- Live API polling on the SPX dashboard (no full-page reload)
- Scenario range slider, flow overlay card, model overlay status
- `scripts/gex_compact_exports.py` retention helper
- Intraday GitHub Actions workflow (`.github/workflows/intraday_exports.yml`)
- Optional ML deps in `requirements-ml.txt`

Future ideas:

- Production UW flow websocket adapter (see `docs/LIVE_FEED.md`)
- Parquet export format for large ML training sets
- Multi-ticker support behind a feature flag when API quota allows
