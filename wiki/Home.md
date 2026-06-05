# Gamma Exposure Tracker (GEX)

Welcome to the **GEX** project wiki. This documentation complements the [README](https://github.com/TheMitchyBoy/GEX/blob/main/README.md) with deeper guides for operators and contributors.

GEX analyzes **dealer gamma exposure** in equity options markets using data from the [Unusual Whales](https://unusualwhales.com/) API. Snapshots are stored as timestamped CSV/JSON exports and power a Flask SPX dashboard, forecasting models, and optional live flow overlays.

## Quick links

| Topic | Page |
|-------|------|
| Install & first run | [Getting Started](Getting-Started) |
| System design | [Architecture](Architecture) |
| Flask dashboard & REST API | [Dashboard & API](Dashboard-and-API) |
| KNN + model overlay | [Forecasting](Forecasting) |
| Environment variables | [Configuration](Configuration) |
| Planned work | [Roadmap](Roadmap) |

## What you can do

- **Fetch & export** SPX GEX via CLI (`main.py`) or scheduled GitHub Actions
- **Visualize** gamma regime, walls, flip, 0DTE movement, and SPX price on the web dashboard
- **Forecast** next-snapshot ΔGEX with weighted KNN, prediction intervals, and optional XGBoost/LSTM overlay
- **Alert** on regime shifts, wall migration, and flip crossings (with optional webhook dispatch)
- **Blend** live option flow from a JSONL feed into forecasts

## Disclaimer

This tool is for **educational and informational purposes only**. GEX analysis reflects assumptions about dealer hedging behavior and is **not financial advice**.
