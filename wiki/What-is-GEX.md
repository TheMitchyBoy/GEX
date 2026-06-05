# What is GEX?

**Gamma** (Γ) measures how an option's **delta** changes when the underlying price moves. Market makers and dealers hedge options by trading the underlying, so **aggregate gamma exposure (GEX)** influences how prices react to flows.

## Regimes

| Total GEX | Regime | Typical market effect |
|-----------|--------|------------------------|
| Positive | **LONG gamma** | Hedging tends to **dampen** moves (buy dips, sell rips) |
| Negative | **SHORT gamma** | Hedging can **amplify** moves (sell dips, buy rips) |
| Near zero | **Neutral** | Limited gamma-driven stabilization |

## Key structural levels

Derived from the strike-level GEX distribution:

- **Call wall** — strike with the largest positive GEX (often cited as resistance where dealers sell into rallies)
- **Put wall** — strike with the largest negative GEX (often cited as support / acceleration zone)
- **Gamma flip** — strike where **cumulative** GEX crosses zero (local regime change above vs below)

## GEX formula (OI-based)

Per-contract notional gamma exposure in **billions of dollars per 1% underlying move**:

```
GEX = spot × gamma × open_interest × 100 × spot × 0.01 × sign
```

- **Calls** → positive sign  
- **Puts** → negative sign  

Unusual Whales strike aggregates use **trade-verified** exposure and may differ from open-interest-only estimates.

## How GEX uses this

1. Pull UW strike/expiration aggregates  
2. Compute walls, flip, term structure (0DTE share, curvature)  
3. Classify regime and feed features into KNN forecasting  
4. Present levels and forecasts on the [Dashboard](Dashboard-and-API)
