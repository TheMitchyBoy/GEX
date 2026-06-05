# Data Quality

Contract-level parsing and filtering run in `gex_core.data_quality.clean_option_data()` when processing option chains. UW strike aggregates bypass per-contract filters but summary exports can still record quality metadata.

## Filter configuration

| Variable | Default | Filter tag |
|----------|---------|------------|
| `GEX_DATA_FILTERS` | `1` | Master switch |
| `GEX_MIN_OPEN_INTEREST` | `1` | `low_oi` |
| `GEX_MIN_GAMMA` | `0` | `non_positive_gamma` |
| `GEX_MAX_STRIKE_DISTANCE_PCT` | `0.35` | `far_otm` |
| `GEX_MAX_IV` | `6.0` | `iv_outlier` |
| `GEX_MAX_BID_ASK_SPREAD_PCT` | `1.0` | `wide_or_crossed_spread` |
| `GEX_DEDUPE_SYMBOLS` | `1` | `duplicate_symbol` |

Always applied when filters are on: `invalid_symbol`, `expired`.

## Export metadata

Summary JSON includes:

- `export_schema_version`  
- `filter_config_hash` — reproducibility fingerprint of active filters  
- `data_quality` — per-step drop counts when available  

## Dashboard trust panel

The **Data Quality / Trust** card scores:

- Strike depth  
- Non-zero strike ratio  
- Top-5 concentration  
- Whether gamma flip lies inside strike range  
- Snapshot age vs intraday cadence  

## Stricter SPX preset

```bash
export GEX_MIN_OPEN_INTEREST=10
export GEX_MAX_STRIKE_DISTANCE_PCT=0.25
export GEX_MAX_BID_ASK_SPREAD_PCT=0.5
```

See also [Configuration](Configuration).
