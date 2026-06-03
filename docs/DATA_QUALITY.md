# Data quality filters

All filters run in `gex_core.data_quality.clean_option_data()` before GEX is calculated. Each step logs how many contracts were removed.

## Master switch

| Variable | Default | Effect |
|----------|---------|--------|
| `GEX_DATA_FILTERS` | `1` | Set to `0` to disable every filter (parse symbols only) |

## Filters

| Variable | Default | Step name | Effect |
|----------|---------|-----------|--------|
| — | — | `invalid_symbol` | Drop rows that fail OCC symbol parse |
| `GEX_MIN_OPEN_INTEREST` | `1` | `low_oi` | Drop zero/trivial open interest |
| `GEX_MIN_GAMMA` | `0` | `non_positive_gamma` | Drop non-positive gamma |
| — | — | `expired` | Drop expirations before today |
| `GEX_MAX_STRIKE_DISTANCE_PCT` | `0.35` | `far_otm` | Drop strikes outside ±35% of spot |
| `GEX_MAX_IV` | `6.0` | `iv_outlier` | Drop IV ≤ 0 or IV > cap (CBOE decimal) |
| `GEX_MAX_BID_ASK_SPREAD_PCT` | `1.0` | `wide_or_crossed_spread` | Drop quoted rows with crossed/wide spreads |
| `GEX_DEDUPE_SYMBOLS` | `1` | `duplicate_symbol` | Keep highest-OI row per symbol |

## Example log line

```
Data quality: 8421/12004 contracts kept (3183 removed: invalid_symbol -12, low_oi -890, far_otm -2100, iv_outlier -181).
```

## Stricter SPX example

```bash
GEX_MIN_OPEN_INTEREST=10
GEX_MAX_STRIKE_DISTANCE_PCT=0.25
GEX_MAX_BID_ASK_SPREAD_PCT=0.5
```

## Future improvements

- Secondary data vendor + OCC OI reconciliation
- Stale quote exclusion via `last_trade_time`
- IV-surface gamma instead of per-row exchange gamma
- Persist per-snapshot `DataQualityReport` in the database for monitoring
