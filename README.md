# Gamma Exposure Tracker (GEX)

A Python tool to analyze dealer gamma exposure (GEX) in equity options markets. GEX is loaded from the **Unusual Whales** API and saved as timestamped CSV exports for dashboards, history, and models.

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
   - `streamlit` and `plotly` - interactive dashboard visualization
   - `Flask` - website dashboard server
   - `yfinance` - historical price data for model training
   - `xgboost` - gradient boosting model support
   - `tensorflow` - LSTM model training
   - `joblib` - model serialization

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

**Limit charts to ±1% around the current price:**
```bash
python main.py --ticker SPX --strike-window-pct 0.01
```

**Run the Streamlit dashboard:**
```bash
streamlit run streamlit_app.py
```

**Run the Flask website dashboard:**
```bash
python web_app.py
```

**Run the live option-flow ingest sample:**
```bash
python live/ingest.py --feed data/flow_sample.jsonl --spot 4800
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
| `--strike-window-pct` | float | 0.01 | Strike window (±1% by default, maximum 0.01). |
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

Each run also writes a machine-readable summary file:

```
{TICKER}_summary_{timestamp}.json
```

### 0DTE surface example

```bash
python main.py --ticker SPX --max-dte 0 --no-show
```

This generates the same-day (0DTE) surface plot and exports the data without opening plot windows.

---

## Dashboard & Website

The repository now includes two SPX-focused interactive interfaces for viewing GEX output:

- **Streamlit dashboard:** `streamlit_app.py`
  - Visualizes the latest SPX CSV exports from `data/exports`
  - Shows interactive heatmaps and 3D scatter plots
  - Displays PNG snapshots from `img/`
- **Flask website:** `web_app.py`
  - Includes a historical timeline selector to revisit older GEX snapshots
  - Leads with SPX gamma-analysis cards for regime, flip distance, wall proximity, gamma balance, and near-term concentration
  - Adds AI-like nearest-neighbor prediction for the next GEX update from current outlook
  - Surfaces similar historical setup matches and shows what happened on the next snapshot

Run the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

Run the Flask dashboard:
```bash
python web_app.py
```

**Data source:** Unusual Whales only (`UW_API_KEY` required). Each run or scheduled refresh writes strike/cumulative GEX to `data/exports/` as timestamped CSV + JSON summary files.

Configuration (environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `UW_API_KEY` | (unset) | Unusual Whales API key (primary source) |
| `GEX_REFRESH_INTERVAL_MINUTES` | `10` | Auto-refresh interval |
| `GEX_DISABLE_SCHEDULER` | (unset) | Set to `1` to disable background refresh |
| `GEX_DATA_FILTERS` | `1` | Set to `0` to skip option filters |
| `GEX_MIN_OPEN_INTEREST` | `1` | Minimum open interest per contract |
| `GEX_MAX_STRIKE_DISTANCE_PCT` | `0.35` | Max strike distance from spot |

See [docs/DATA_QUALITY.md](docs/DATA_QUALITY.md) and [config/spx.env.example](config/spx.env.example) for stricter SPX filter presets.

Manual CSV refresh:

```bash
export UW_API_KEY=your-key
python scripts/gex_refresh.py --force
```

Backfill recent UW market snapshots for regime-relevant prediction history:

```bash
python scripts/gex_refresh.py --tickers SPX --backfill-days 7
```

Docker:

```bash
docker compose up web
```

## Live Option Flow Signal

A live ingest pipeline is included under `live/` that consumes JSON-lines option flow events and computes real-time GEX strike signals.

Example feed usage:
```bash
python live/ingest.py --feed data/flow_sample.jsonl --spot 4800
```

The live flow aggregator computes:
- per-strike GEX deltas from traded option flow
- decayed historical signal weightings
- buy/sell flow imbalance
- aggressiveness-adjusted score and direction

## Model Training

The project includes model training scripts for GEX prediction. The default
delta-GEX model trains on the most recent 7 calendar days so the overlay stays
aligned with the current market regime:

- `scripts/train_gex_model.py` — trains a next-snapshot ΔGEX regressor from recent historical exports
- `scripts/train_gex_lstm.py` — trains an LSTM on sequential export feature history

Example training command:
```bash
python scripts/train_gex_model.py --ticker SPX --lookback-days 7
python scripts/train_gex_lstm.py --ticker SPX --seq-len 8 --epochs 50 --batch-size 16
```

## Scheduled Daily Exports

A GitHub Actions workflow is configured to run `main.py` daily for `SPX` and commit updated CSV exports to `data/exports`.

Workflow path:
- `.github/workflows/daily_exports.yml`

This keeps the repository populated with fresh daily gamma exposure exports for models and dashboards.

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

### Unusual Whales

GEX by strike is loaded from the Unusual Whales API (`/api/stock/{ticker}/greek-exposure/strike`). Snapshots are written to `data/exports/` as:

- `{TICKER}_gex_by_strike_{timestamp}.csv`
- `{TICKER}_cumulative_gex_{timestamp}.csv`
- `{TICKER}_summary_{timestamp}.json`

Set `UW_API_KEY` in your environment (or Cursor secrets) before running the CLI, refresh script, or web dashboard.

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

## Developer Quick Start

```bash
pip install -r requirements.txt -r requirements-dev.txt
export UW_API_KEY=your-key          # optional; enables UW primary source
python main.py --ticker SPX --no-show
python web_app.py                   # Flask dashboard
streamlit run streamlit_app.py
python scripts/gex_refresh.py --force
python scripts/backtest_gex_prediction.py --ticker SPX
pytest
```

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements.
