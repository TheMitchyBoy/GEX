# Getting Started

## Requirements

- Python **3.11+**
- [Unusual Whales](https://unusualwhales.com/) API key: `UW_API_KEY`

## Installation

```bash
git clone https://github.com/TheMitchyBoy/GEX.git
cd GEX
pip install -r requirements.txt
export UW_API_KEY=your-key-here
```

Development:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

Optional ML training (XGBoost / LSTM):

```bash
pip install -r requirements-ml.txt
```

## First CLI run

```bash
python main.py --ticker SPX
python main.py --ticker SPX --no-show          # headless (CI / cron)
python main.py --ticker SPX --market-date 2026-06-01
```

Exports land in `data/exports/` with names like `SPX_gex_by_strike_2026-06-05_143000.csv`.

## Web dashboard

```bash
python web_app.py
# Open http://localhost:8501/ticker/SPX
```

Docker:

```bash
docker compose up web
```

## Manual refresh

```bash
python scripts/gex_refresh.py --force
python scripts/gex_refresh.py --tickers SPX --backfill-days 7
```

## Backfill history for forecasting

More snapshots improve KNN and walk-forward backtests. Use UW historical dates or wait for scheduled Actions to populate `data/exports/`.

```bash
python scripts/backtest_gex_prediction.py --ticker SPX
```

## Next steps

- [Configuration](Configuration) — environment variables  
- [Dashboard & API](Dashboard-and-API) — UI features and `/health`  
- [Forecasting](Forecasting) — how predictions work  
- [Roadmap](Roadmap) — recent and planned improvements  
