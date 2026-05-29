# Gamma Exposure Tracker (GEX)

A Python tool to analyze dealer gamma exposure (GEX) in equity options markets. This script scrapes real-time options data from the CBOE website, calculates aggregate gamma exposure across all dealers, and visualizes gamma dynamics across strikes and expirations.

## Table of Contents

- [What is Gamma Exposure?](#what-is-gamma-exposure)
- [Understanding GEX](#understanding-gex)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Gamma Calculation Formula](#gamma-calculation-formula)
- [Output Interpretation](#output-interpretation)
- [Visualizations](#visualizations)
- [Data Source](#data-source)
- [Requirements](#requirements)

---

## What is Gamma Exposure?

**Gamma** is a second-order option Greeks metric that measures how an option's delta changes as the underlying price moves. In simpler terms:
- **Delta** tells you how much an option price moves for a 1% move in the underlying
- **Gamma** tells you how much the delta itself changes for that 1% move

### Why Dealers Matter

Options dealers typically hedge their positions by buying/selling the underlying stock. When they accumulate options positions, their hedging needs create market dynamics:
- **Long Gamma Dealers**: Profit from volatility, tend to buy dips and sell rallies (stabilizing effect)
- **Short Gamma Dealers**: Lose from volatility, forced to sell dips and buy rallies (destabilizing effect)

Aggregate dealer gamma exposure indicates the market regime and potential price levels where dealer hedging behavior becomes significant.

---

## Understanding GEX

### Total Notional GEX

The **Total Notional GEX** represents the aggregate gamma exposure (in billions of dollars) of all dealers across all options expiring on that day and beyond.

**Interpretation:**

| Total GEX | Regime | Market Dynamics |
|-----------|--------|-----------------|
| **Positive (e.g., +$50 Bn)** | LONG gamma | Dealers are long gamma; they profit from volatility and hedging creates stabilizing pressure |
| **Negative (e.g., -$50 Bn)** | SHORT gamma | Dealers are short gamma; they lose from volatility and hedging creates destabilizing pressure |
| **Near Zero** | Neutral | Balanced gamma exposure; limited gamma-driven hedging flow |

### Example Output

```bash
Total notional GEX: $-38.1193 Bn
Net gamma regime: SHORT gamma (-38.119 Bn$ / %)
Estimated call wall: strike 4800 (45.123)
Estimated put wall: strike 4750 (-38.105)
```

**What this means:**
- The market is in a **SHORT gamma regime** (dealers are short gamma overall)
- Dealers may be forced to sell into rallies and buy into dips
- The largest positive gamma exposure (call wall) is at strike 4800
- The largest negative gamma exposure (put wall) is at strike 4750

---

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository** or download as ZIP file:
   ```bash
   git clone https://github.com/TheMitchyBoy/GEX.git
   cd GEX
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This installs:
   - `pandas` - Data manipulation and analysis
   - `requests` - HTTP client for fetching CBOE data
   - `matplotlib` - Plotting and visualization

---

## Quick Start

### Basic Usage

Run the script and enter a ticker symbol:

```bash
python main.py
Enter desired ticker: SPX
```

Or provide the ticker as a command-line argument:

```bash
python main.py --ticker SPX
```

### Common Use Cases

**Force refresh data (ignore cache):**
```bash
python main.py --ticker SPX --refresh
```

**Only include options expiring within 60 days:**
```bash
python main.py --ticker SPX --max-dte 60
```

**Generate plots without displaying them:**
```bash
python main.py --ticker SPX --no-show
```

**Wider strike window (±20% around current price):**
```bash
python main.py --ticker SPX --strike-window-pct 0.20
```

---

## Usage

### Command-Line Arguments

```bash
python main.py --ticker SYMBOL [options]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ticker` | str | (prompted) | Underlying ticker symbol (e.g., SPX, XES) |
| `--refresh` | flag | False | Force refresh data, ignore cache |
| `--cache-ttl-minutes` | int | 15 | Cache validity duration in minutes |
| `--no-show` | flag | False | Don't display plots in windows |
| `--no-save` | flag | False | Don't save plots to disk |
| `--outdir` | str | `img/` | Output directory for plots |
| `--top-n` | int | 5 | Number of top strikes to highlight |
| `--strike-window-pct` | float | 0.15 | Strike window (±15% by default) |
| `--max-dte` | int | 365 | Maximum days-to-expiration to include. Use 0 for same-day (0DTE) only. |
| `--no-export-csv` | flag | False | Disable CSV exports |
| `--export-dir` | str | `data/exports/` | CSV export directory |

### Example Session

```bash
$ python main.py --ticker SPX --top-n 3 --strike-window-pct 0.20

Total notional GEX: $-38.1193 Bn
Net gamma regime: SHORT gamma (-38.119 Bn$ / %)
Estimated call wall: strike 4800 (45.123)
Estimated put wall: strike 4750 (-38.105)

Top 3 positive GEX strikes (Bn$ / %):
  Strike 4800: 45.123
  Strike 4850: 38.456
  Strike 4750: 32.789

Top 3 negative GEX strikes (Bn$ / %):
  Strike 4700: -42.100
  Strike 4650: -35.200
  Strike 4600: -28.900

Estimated gamma flip strike: 4800.50 (confidence: high)
Saved plot: img/SPX_gex_by_strike_2025-05-28_143025.png
Saved plot: img/SPX_gex_by_expiration_2025-05-28_143026.png
Saved plot: img/SPX_gex_surface_2025-05-28_143027.png
Saved CSV exports to: data/exports
```

### 0DTE surface example

```bash
python main.py --ticker SPX --max-dte 0 --no-show
```

This generates the same-day (0DTE) surface plot and exports the data without opening plot windows.

---

## Gamma Calculation Formula

### Per-Contract GEX Calculation

For each option contract, the notional gamma exposure is calculated as:

```
GEX = Spot Price × Gamma × Open Interest × Contract Size × Spot Price × 0.01
```

Where:
- **Spot Price** - Current underlying price (appears twice to convert to notional dollar amount)
- **Gamma** - Option gamma (Greek) provided by CBOE
- **Open Interest** - Number of contracts outstanding
- **Contract Size** - 100 shares per standard equity option contract
- **0.01 Factor** - Converts 1% gamma into decimal format

### Sign Convention

The formula accounts for dealer hedging behavior:
- **Call options**: Positive gamma (dealers long calls)
- **Put options**: Negative gamma (dealers short puts)

```python
GEX_Calls = +1 × (formula above)   # Positive contribution
GEX_Puts  = -1 × (formula above)   # Negative contribution
```

### Total GEX

**Total Notional GEX** = Sum of all individual option GEX values

```
Total GEX = Σ(GEX_Calls) + Σ(GEX_Puts)
```

### Interpretation of the Formula

The GEX figure represents **dollars of gamma exposure per 1% move in the underlying**:
- A total GEX of **-$38 Billion** means if the underlying moves 1%, dealers' delta hedge requirements change by approximately $38 billion
- In a negative GEX regime, dealers are forced to buy as the market falls and sell as it rises

---

## Output Interpretation

### 1. Total Notional GEX

Aggregate gamma exposure metric for overall market regime.

```
Total notional GEX: $-38.1193 Bn
```

- **Positive**: Market in LONG gamma regime (stabilizing)
- **Negative**: Market in SHORT gamma regime (destabilizing)

### 2. Net Gamma Regime

Summary of the gamma environment and concentration:

```
Net gamma regime: SHORT gamma (-38.119 Bn$ / %)
Estimated call wall: strike 4800 (45.123)
Estimated put wall: strike 4750 (-38.105)
```

- **Call Wall**: Strike with the highest positive GEX (maximum dealer long call exposure)
- **Put Wall**: Strike with the lowest negative GEX (maximum dealer short put exposure)
- These levels represent critical price support/resistance zones where gamma flows activate

### 3. Top GEX Levels

The most significant gamma concentrations:

```
Top 5 positive GEX strikes (Bn$ / %):
  Strike 4800: 45.123
  Strike 4850: 38.456
  ...

Top 5 negative GEX strikes (Bn$ / %):
  Strike 4700: -42.100
  Strike 4650: -35.200
  ...
```

**Trading insight**: These levels often act as support/resistance where dealer gamma hedging becomes active.

### 4. Gamma Flip Level

The estimated strike where cumulative gamma exposure crosses zero:

```
Estimated gamma flip strike: 4800.50 (confidence: high)
```

- **High confidence**: Sharp, reliable transition
- **Medium confidence**: Moderate transition
- **Low confidence**: Gradual transition, less reliable

**Trading use**: Gamma flips indicate where dealer hedging behavior reverses.

---

## Visualizations

The tool generates three types of visualizations:

### 1. GEX by Strike (`gamma_by_strike.png`)

<p align="center">
  <img 
    src="img/gamma_by_strike.png"
    alt="GEX by Strike Price"
    width="80%"
  >
</p>

**What it shows:**
- Gamma exposure distribution across strike prices
- **Green bars**: Positive GEX (long gamma)
- **Pink bars**: Negative GEX (short gamma)
- Focused window around current spot price (±15% by default, configurable)

**How to interpret:**
- Tall green bars = price levels with concentrated bullish gamma
- Tall pink bars = price levels with concentrated bearish gamma
- Gaps in bars = potential price targets with low gamma support/resistance

### 2. GEX by Expiration (`gamma_by_expiration.png`)

<p align="center">
  <img 
    src="img/gamma_by_expiration.png"
    alt="GEX by Expiration Date"
    width="80%"
  >
</p>

**What it shows:**
- Gamma exposure distribution across expiration dates (term structure)
- Bar height represents total gamma at each expiration

**How to interpret:**
- Tall bars near near-term expirations often mean active trading/hedging
- Peak exposures align with major expirations (weeklies, monthlies, quarterlies)
- Helps identify which expiration cycles drive market dynamics

### 3. GEX Surface (`surface.png`)

<p align="center">
  <img 
    src="img/surface.png"
    alt="3D GEX Surface"
    width="80%"
  >
</p>

**What it shows:**
- 3D visualization combining strikes (X), expirations (Y), and GEX magnitude (Z)
- **Red regions**: Positive GEX (long gamma)
- **Blue regions**: Negative GEX (short gamma)

**How to interpret:**
- Hot spots (tall peaks/valleys) = critical gamma levels
- Smooth surface = evenly distributed gamma
- Steep transitions = sharp changes in hedging behavior across price levels
- Helps identify gamma clusters across both dimensions simultaneously

---

## Data Source

### CBOE API

Data is fetched from the **Chicago Board Options Exchange (CBOE)** delayed quotes API:

```
https://cdn.cboe.com/api/global/delayed_quotes/options/{TICKER}.json
```

**Data Included:**
- Current underlying price (delayed by ~15 minutes)
- All active options contracts for the ticker
- Option Greeks: delta, gamma, theta, vega, rho
- Open interest and volume
- Option prices (bid/ask)

**Caching:**
- Data is cached locally (default: 15 minutes) to minimize API calls
- Use `--refresh` flag to force fresh data
- Cache files stored in `data/` directory

### Delay & Freshness

- **CBOE data**: ~15 minute delay (market data rules)
- **Cache validity**: 15 minutes by default (configurable)
- Total data age: Up to 30 minutes old in normal operation

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.2.2+ | DataFrame operations, data aggregation |
| `requests` | 2.27.1+ | HTTP requests to CBOE API |
| `matplotlib` | 3.8.0+ | Plotting and visualization |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Disclaimer

This tool is for educational and informational purposes. Gamma exposure analysis is one of many factors in options market analysis. The GEX calculations and interpretations are based on assumptions about dealer behavior and should not be considered financial advice. Always conduct your own research before making trading decisions.

---

## License

[Include your license information here if applicable]

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements.
