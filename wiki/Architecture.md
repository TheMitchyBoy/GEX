# Architecture

## Data flow

```
Unusual Whales API
       │
       ▼
  main.py / gex_core.refresh
       │
       ▼
  data/exports/          ← timestamped CSV + JSON snapshots
       │                      (optional SQLite index: data/gex_index.db)
       ├──────────────────────────────────────┐
       ▼                                      ▼
  gex_core.history                    gex_core.predict
  (timeline, features)                (KNN + optional model overlay)
       │                                      │
       └──────────────┬───────────────────────┘
                      ▼
              web_app.py (Flask SPX dashboard)
                      │
              live/ (optional JSONL flow overlay)
```

## Persistence model

| Layer | Location | Purpose |
|-------|----------|---------|
| Snapshots | `data/exports/` | Source of truth: strike, cumulative, expiration, surface, summary JSON |
| Index | `data/gex_index.db` | Fast latest-timestamp and catalog queries |
| Models | `models/{TICKER}/` | Trained overlay + `manifest.json` |
| Flow state | `GEX_FLOW_FEED` path | JSONL events for intraday overlay |

Each refresh writes a **matched file set** sharing one timestamp suffix: `YYYY-MM-DD_HHMMSS`.

## Core packages

| Path | Role |
|------|------|
| `gex_core/uw_loader.py` | Unusual Whales HTTP client |
| `gex_core/history.py` | Build chronologic snapshot list from exports |
| `gex_core/features.py` | Feature vectors, term structure, surface similarity |
| `gex_core/predict.py` | Weighted KNN, intervals, model blend |
| `gex_core/intelligence.py` | Alerts, confluence, scenario simulation, panels |
| `gex_core/ai_analyst.py` | Rule-based dealer narrative (+ optional OpenAI polish) |
| `gex_core/storage.py` | SQLite export index |
| `live/` | Flow ingest and strike-level ΔGEX |

## UIs

| UI | Entry | Notes |
|----|-------|-------|
| **Flask** (primary) | `web_app.py` / `wsgi.py` | SPX dashboard, scheduler, REST API |
| **Streamlit** | `streamlit_app.py` | Explorer / alternate view |
| **CLI** | `main.py` | Fetch, print, plot, export |

## Automation

- **Daily exports** — `.github/workflows/daily_exports.yml`  
- **Intraday exports** — `.github/workflows/intraday_exports.yml` (weekdays)  
- **CI tests** — `.github/workflows/ci.yml` (includes backtest quality gates)  
- **Wiki publish** — `.github/workflows/publish-wiki.yml` (syncs `wiki/` → GitHub Wiki)
