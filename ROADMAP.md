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

### Deepen training data via a higher UW API tier

The prediction overlays are currently constrained by **training-data volume**, not
algorithm choice. Both trainers (`scripts/train_gex_model.py`,
`scripts/train_gex_lstm.py`) build datasets only from snapshots already in
`data/exports/`, and historical backfill is gated by the UW plan's look-back
window. The current key returns `403` for `greek-exposure/strike` dates older
than ~2 weeks (≈8–10 trading days), which yields only ~8–10 daily rows — below
the XGBoost threshold (≥10 train rows after split) and far too few for the LSTM
(`seq_len=8`).

UW differentiates API subscriptions by **historical look-back depth** and
**daily request limits** (see the UW changelog "API subscriptions: increased
historical look back + daily limits"). Upgrading would help by:

- **Deeper look-back** → backfill many more EOD snapshots (e.g. ~250–500 rows for
  1–2 years), making the XGBoost/LSTM overlays genuinely trainable with real
  walk-forward CV instead of falling back to linear on ~10 points. *(Biggest lever.)*
- **Higher daily/rate limits** → removes the `429`s behind transient
  "Snapshot refresh failed" errors and lets the 1-minute scheduler + bulk
  backfill run reliably.

Caveats / required work:
- More data improves *trainability and statistical robustness*, not guaranteed
  predictive accuracy — next-day ΔGEX is inherently noisy.
- The `date=` endpoint returns **EOD** granularity only. Intraday-historical
  density (the most valuable kind) would require a **new loader path** for a UW
  time-series/intraday endpoint if the tier exposes one.
- Verify the exact look-back depth and daily quota of the target tier on the UW
  pricing page before upgrading — that depth determines the magnitude of benefit.
- Follow-up after upgrade: backfill (`scripts/gex_refresh.py --backfill-days N
  --force`) then retrain (`scripts/train_gex_model.py --lookback-days 0`, and
  `scripts/train_gex_lstm.py` once enough rows exist).

See [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) for a prioritized engineering
backlog (security, correctness, reliability, performance, testing, ops, UX).
