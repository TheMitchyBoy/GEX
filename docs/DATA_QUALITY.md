# Options Data Quality

## Built-in filters (env-configurable)

| Variable | Default | Effect |
|----------|---------|--------|
| `GEX_MIN_OPEN_INTEREST` | `1` | Drop contracts with zero/trivial OI |
| `GEX_MIN_GAMMA` | `0` | Drop non-positive gamma |
| `GEX_MAX_IV` | `6.0` | Drop implied vol outliers (CBOE decimal IV) |
| `GEX_MAX_BID_ASK_SPREAD_PCT` | `1.0` | Drop quoted contracts with crossed/wide spreads |
| `GEX_MAX_STRIKE_DISTANCE_PCT` | `0.35` | Drop ultra-far OTM wings (noise vs signal) |

Filters run in `gex_core.clean_option_data()` before GEX is computed.

## Ideas to improve quality further

### Data sources
- **Primary + fallback**: CBOE delayed feed (current) + OCC open-interest file for OI reconciliation.
- **Vendor upgrade**: Polygon, ORATS, or ThetaData for consolidated chains, corrected Greeks, and timestamps.
- **Index specifics**: Merge SPX + SPXW (weekly) chains; exclude ES options when analyzing cash SPX.

### Microstructure hygiene
- Weight GEX by **volume × OI** or use OI only above a rolling percentile (e.g. 90th).
- Exclude **penny options** and strikes with `last_trade_time` older than N hours.
- Cap single-contract GEX contribution to reduce bad ticks.
- Use **mid price** (bid/ask) to recompute gamma via BS when exchange gamma is stale.

### Greek modeling
- Recompute gamma from IV surface (SVI/SABR) instead of vendor per-contract gamma.
- Separate **customer vs firm** OI if available; dealer GEX assumptions differ by book.
- Add **vanna/charm** consistency checks when charm and gamma disagree in sign/magnitude.

### Operational
- Store `data_quality_report` fields per snapshot (rows in/out, filter counts).
- Alert when filtered % exceeds threshold (possible feed break).
- Compare total GEX vs prior snapshot; flag >3σ moves for manual review.
