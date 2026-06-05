# Dashboard & API

The **Flask** app (`web_app.py`) is the primary SPX interface. The app is **SPX-only** by policy (`gex_core.tickers`).

## Running

```bash
python web_app.py
# Default: http://0.0.0.0:8501
```

Production (gunicorn):

```bash
gunicorn --bind 0.0.0.0:8080 wsgi:app
```

## Main routes

| Route | Description |
|-------|-------------|
| `/` | SPX command center / watchlist cards |
| `/ticker/SPX` | Full gamma dashboard |
| `/widget/SPX` | Compact embed (`?theme=dark&compact=1`) |
| `/health` | JSON operational status |
| `/api/latest-summary` | JSON payload for polling |
| `/api/signals` | Alerts, confluence, probabilities |
| `/api/watchlist` | Watchlist rows |
| `/exports/<file>` | Download export files |

## Dashboard features

- **Gamma regime** card — total GEX, walls, flip distance  
- **GEX profile charts** — CSV snapshot + optional live UW overlay  
- **0DTE movement** — same-day prior snapshot comparison  
- **SPX price chart** — Yahoo Finance with snapshot fallback  
- **Timeline & cumulative GEX**  
- **Prediction** — ΔGEX, intervals, neighbor typical error  
- **Scenario simulator** — spot shift slider (-3% to +3%)  
- **Alert engine** — rule-based; manual or auto webhook dispatch  
- **Confluence score** — structure + model + flow alignment  
- **Live polling** — `/api/latest-summary` updates status banner without full reload  

## Health endpoint

```bash
curl -s http://localhost:8080/health | jq
```

Example fields:

- `healthy` — history exists and export not stale  
- `export_age_minutes`, `latest_export_ts`  
- `history_depth`  
- `uw_api_configured`, `scheduler_enabled`  
- `model_overlay_active`, `model_training_rows`  
- `alert_webhook_configured`, `alert_auto_dispatch`  

Returns **503** when unhealthy (no history or stale export).

## Force refresh

```
/ticker/SPX?force_refresh=1
```

Triggers CSV export refresh (if `UW_API_KEY` set) and live UW cache refresh.

## Embed widget

```html
<iframe src="https://your-host/widget/SPX?compact=1" width="320" height="200"></iframe>
```

## Streamlit

Alternate explorer:

```bash
streamlit run streamlit_app.py
```

Flask remains the recommended production dashboard.
