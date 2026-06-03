# Data quality

Contracts are parsed and filtered in `gex_core.data_quality.clean_option_data()` before GEX runs.

## Configuration

| Variable | Default | Filter |
|----------|---------|--------|
| `GEX_DATA_FILTERS` | `1` | Master switch (`0` = parse only) |
| `GEX_MIN_OPEN_INTEREST` | `1` | `low_oi` |
| `GEX_MIN_GAMMA` | `0` | `non_positive_gamma` |
| `GEX_MAX_STRIKE_DISTANCE_PCT` | `0.35` | `far_otm` |
| `GEX_MAX_IV` | `6.0` | `iv_outlier` |
| `GEX_MAX_BID_ASK_SPREAD_PCT` | `1.0` | `wide_or_crossed_spread` |
| `GEX_DEDUPE_SYMBOLS` | `1` | `duplicate_symbol` |

Always applied when filters are on: `invalid_symbol`, `expired`.

## Output

- Console: one summary line per run
- Exports: `summary.json` → `data_quality` object with per-step counts

## Stricter SPX example

```bash
export GEX_MIN_OPEN_INTEREST=10
export GEX_MAX_STRIKE_DISTANCE_PCT=0.25
export GEX_MAX_BID_ASK_SPREAD_PCT=0.5
```
