# Gamma Exposure Tracker (GEX)

Analyze dealer gamma exposure (GEX) in equity options markets. Data is loaded from the **Unusual Whales** API and saved as timestamped CSV/JSON exports for dashboards, history, and forecasting models.

## What is GEX?

**Gamma** measures how an option's delta changes as the underlying moves. Dealers hedge options positions by trading the underlying, so aggregate gamma exposure shapes how the market responds to price moves:

| Total GEX | Regime | Effect |
|-----------|--------|--------|
| Positive | LONG gamma | Hedging tends to dampen moves (buy dips, sell rips) |
| Negative | SHORT gamma | Hedging can amplify moves (sell dips, buy rips) |
| Near zero | Neutral | Limited gamma-driven flow |

Key levels derived from the strike distribution:

- **Call wall** — strike with the largest positive GEX
- **Put wall** — strike with the largest negative GEX
- **Gamma flip** — strike where cumulative GEX crosses zero

## Architecture

```
Unusual Whales API
       │
       ▼
  main.py / gex_core.refresh  ──►  data/exports/  (CSV + JSON snapshots)
       │                                    │
       ├─ streamlit_app.py                  ├─ gex_core.history  (timeline)
       └─ web_app.py (Flask)                ├─ gex_core.predict  (KNN forecast)
                                            └─ scripts/train_*   (model overlay)
```

Persistence is **file-based** with an optional **SQLite index** (`data/gex_index.db`) for fast history lookups. Each refresh writes a matched set of files sharing a timestamp suffix under `data/exports/`.

## Installation

**Requirements:** Python 3.11+, `UW_API_KEY`

```bash
git clone https://github.com/TheMitchyBoy/GEX.git
cd GEX
pip install -r requirements.txt
export UW_API_KEY=your-key
```

For development:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

## Quick start

### CLI

```bash
python main.py --ticker SPX
python main.py --ticker SPX --no-show          # headless (CI / cron)
python main.py --ticker SPX --market-date 2026-06-01  # historical UW date
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--ticker` | (prompted) | Symbol, e.g. `SPX` |
| `--no-show` | off | Skip plot windows |
| `--no-save` | off | Skip PNG plots in `img/` |
| `--strike-window-pct` | `0.01` | Chart strike window (±1%) |
| `--top-n` | `5` | Top strikes to print |
| `--no-export-csv` | off | Skip CSV/JSON export |
| `--export-dir` | `data/exports/` | Export directory |
| `--uw-key` | env | Override `UW_API_KEY` |
| `--market-date` | today | Historical date `YYYY-MM-DD` |

### Dashboards

**Flask (primary):**

```bash
python web_app.py
# or: docker compose up web
```

**Streamlit:**

```bash
streamlit run streamlit_app.py
```

### Manual refresh

```bash
python scripts/gex_refresh.py --force
python scripts/gex_refresh.py --tickers SPX --backfill-days 7
```

## GEX formula

Per-contract notional gamma exposure (Bn$ per 1% underlying move):

```
GEX = spot × gamma × open_interest × 100 × spot × 0.01 × sign
```

Calls contribute positive GEX; puts contribute negative GEX. Unusual Whales strike aggregates use trade-verified exposure and may differ from open-interest-only estimates.

## Forecasting

The prediction stack (`gex_core.predict`) combines:

- **Weighted KNN** on regime feature vectors with empirical prediction intervals
- **Calibrated confidence** from walk-forward sign accuracy
- **Market context** — realized vol, spot returns, 0DTE ratio (VIX when online)
- **Regime-conditional blending** with optional trained-model overlay (XGBoost / LSTM)
- **Structural attribution** — spot vs residual ΔGEX split (`gex_core.structural`)

Train models:

```bash
python scripts/train_gex_model.py --ticker SPX --lookback-days 7
python scripts/train_gex_lstm.py --ticker SPX --seq-len 8 --epochs 50
python scripts/backtest_gex_prediction.py --ticker SPX
```

## Live option flow

The `live/` package ingests JSON-lines flow events and computes real-time strike-level GEX deltas:

```bash
python live/ingest.py --feed data/flow_sample.jsonl --spot 4800
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `UW_API_KEY` | (required) | Unusual Whales API key |
| `GEX_REFRESH_INTERVAL_MINUTES` | `1` | Web dashboard auto-refresh |
| `GEX_DISABLE_SCHEDULER` | off | Set `1` to disable background refresh |
| `GEX_DATA_FILTERS` | `1` | Set `0` to skip option filters |
| `GEX_MIN_OPEN_INTEREST` | `1` | Minimum OI per contract |
| `GEX_MAX_STRIKE_DISTANCE_PCT` | `0.35` | Max strike distance from spot |
| `GEX_FLOW_FEED` | `data/flow_sample.jsonl` | JSONL flow file for live overlay |
| `GEX_ALERT_WEBHOOK_URL` | off | Webhook URL for alert dispatch |
| `GEX_ALERT_AUTO_DISPATCH` | off | Auto-dispatch high-severity alerts |
| `GEX_INDEX_DB` | `data/gex_index.db` | SQLite export index path |

See [docs/DATA_QUALITY.md](docs/DATA_QUALITY.md), [docs/LIVE_FEED.md](docs/LIVE_FEED.md), and [config/spx.env.example](config/spx.env.example).

Train ML overlays (optional):

```bash
pip install -r requirements-ml.txt
python scripts/train_gex_model.py --ticker SPX
```

## Health & ops

```bash
curl -s localhost:8080/health | jq
```

Compact aged strike CSVs (keeps summaries + cumulative):

```bash
python scripts/gex_compact_exports.py --ticker SPX --keep-full-days 14
```

## Scheduled exports

- Daily: `.github/workflows/daily_exports.yml`
- Intraday (weekdays): `.github/workflows/intraday_exports.yml`

See [ROADMAP.md](ROADMAP.md) for planned work.

## Documentation

- **[GitHub Wiki](https://github.com/TheMitchyBoy/GEX/wiki)** — full project guide (auto-published from [`wiki/`](wiki/))
- [Getting started (wiki)](https://github.com/TheMitchyBoy/GEX/wiki/Getting-Started)
- [Roadmap (wiki)](https://github.com/TheMitchyBoy/GEX/wiki/Roadmap)

## Project layout

| Path | Role |
|------|------|
| `main.py` | CLI entry — fetch, print, plot, export |
| `web_app.py` | Flask SPX dashboard |
| `streamlit_app.py` | Streamlit explorer |
| `gex_core/` | Core library (fetch, history, features, predict) |
| `live/` | Real-time flow ingest |
| `scripts/` | Refresh, training, backtest, compaction |
| `gex_core/storage.py` | SQLite export index |
| `data/exports/` | Timestamped snapshot store |
| `models/` | Trained overlay models + manifests |

## Disclaimer

This tool is for educational and informational purposes only. GEX analysis reflects assumptions about dealer hedging and is not financial advice.
