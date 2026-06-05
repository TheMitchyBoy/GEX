# Roadmap

## Recently shipped

| Feature | Description |
|---------|-------------|
| SQLite export index | `data/gex_index.db` for fast timestamp catalog |
| Export metadata | `export_schema_version`, `filter_config_hash` in summary JSON |
| History cache | LRU cache for `build_history()` |
| CI backtest gates | Walk-forward sign-accuracy floor in GitHub Actions |
| `/health` endpoint | Operational JSON status |
| Alert configuration | Env-driven thresholds + auto webhook with cooldown |
| Dashboard polling | Live status via `/api/latest-summary` |
| Scenario slider | Interactive spot-shift simulation |
| Flow overlay card | Visible when `GEX_FLOW_FEED` has events |
| Model overlay status | Accountability panel shows active/inactive overlay |
| Export compaction | `scripts/gex_compact_exports.py` |
| Intraday Actions | Weekday 30-minute export workflow |
| Optional ML deps | `requirements-ml.txt` (TensorFlow, XGBoost) |
| GitHub Wiki | Auto-published from `wiki/` directory |

## In progress / near term

- Richer intraday history from scheduled workflows (improves KNN and backtest reliability)
- Documented production flow adapters ([Live Flow](Live-Flow))

## Future ideas

| Item | Rationale |
|------|-----------|
| **UW flow websocket adapter** | Real-time flow without manual JSONL tailing |
| **Parquet export format** | Faster ML training at scale |
| **Multi-ticker mode** | Re-enable SPY/NDX behind feature flag when API quota allows |
| **Strike × DTE heatmap** | Visualize surface evolution beyond 1D profiles |
| **Event calendar overlay** | Tag FOMC/CPI days on timeline |
| **Parity: Streamlit vs Flask** | Single view-model layer or deprecate Streamlit |

## Contributing

1. Fork and branch from `main`  
2. Run `pytest` and respect CI backtest gates  
3. Update `wiki/` pages if behavior or configuration changes  
4. Open a PR with a clear description of forecasting/UI impact  

## Links

- [Repository](https://github.com/TheMitchyBoy/GEX)  
- [Issues](https://github.com/TheMitchyBoy/GEX/issues)  
- [Home](Home)  
