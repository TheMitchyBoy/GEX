# Gamma Exposure Tracker (GEX)

Analyze dealer gamma exposure (GEX) in equity options markets. Data is loaded from the **Unusual Whales** API and saved as timestamped CSV/JSON exports for dashboards, history, and forecasting models.

## What is GEX?

**Gamma** measures how an option's delta changes as the underlying moves. Dealers hedge options positions by trading the underlying, so aggregate gamma exposure shapes how the market responds to price moves:

| Total GEX | Regime | Effect |
|-----------|--------|--------|
| Positive | LONG gamma | Hedging tends to dampen moves (buy dips, sell rips) |
| Negative | SHORT gamma | Hedging can amplify moves (sell dips, buy rips) |
| Near zero | Neutral | Limited gamma-driven flow |

Key levels derived from the strike distribution:

- **Call wall** — strike with the largest positive GEX
- **Put wall** — strike with the largest negative GEX
- **Gamma flip** — strike where cumulative GEX crosses zero

## Architecture

```
Unusual Whales API
       │
       ▼
  main.py / gex_core.refresh  ──►  data/exports/  (CSV + JSON snapshots)
       │                                    │
       ├─ streamlit_app.py                  ├─ gex_core.history  (timeline)
       └─ web_app.py (Flask)                ├─ gex_core.predict  (KNN forecast)
                                            └─ scripts/train_*   (model overlay)
```

Persistence is **file-based** with an optional **SQLite index** (`data/gex_index.db`) for fast history lookups. Each refresh writes a matched set of files sharing a timestamp suffix under `data/exports/`.

## Installation

**Requirements:** Python 3.11+, `UW_API_KEY`

```bash
git clone https://github.com/TheMitchyBoy/GEX.git
cd GEX
pip install -r requirements.txt
export UW_API_KEY=your-key
```

For development:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

## Quick start

### CLI

```bash
python main.py --ticker SPX
python main.py --ticker SPX --no-show          # headless (CI / cron)
python main.py --ticker SPX --market-date 2026-06-01  # historical UW date
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--ticker` | (prompted) | Symbol, e.g. `SPX` |
| `--no-show` | off | Skip plot windows |
| `--no-save` | off | Skip PNG plots in `img/` |
| `--strike-window-pct` | `0.01` | Chart strike window (±1%) |
| `--top-n` | `5` | Top strikes to print |
| `--no-export-csv` | off | Skip CSV/JSON export |
| `--export-dir` | `data/exports/` | Export directory |
| `--uw-key` | env | Override `UW_API_KEY` |
| `--market-date` | today | Historical date `YYYY-MM-DD` |

### Dashboards

**Flask (primary):**

```bash
python web_app.py
# or: docker compose up web
```

Open **http://localhost:5000/** for the gamma auto-trader. **http://localhost:5000/trade** is the Webull quick-trade desk: live SPY 0DTE quotes, entry/exit condition signals, and one-click limit orders (passive / mid / smart / aggressive). Requires `GEX_WEBULL_*` credentials and `GEX_TRADER_PAPER=0` for live execution; orders need `live_confirm: true` in the API body.

**Streamlit:**

```bash
streamlit run streamlit_app.py
```

### Manual refresh

```bash
python scripts/gex_refresh.py --force
python scripts/gex_refresh.py --tickers SPX --backfill-days 7
python scripts/gex_refresh.py --tickers SPX --intraday-days 90 --interval-minutes 2
python scripts/gex_backfill_intraday.py --intraday-days 90 --daily-days 90 --interval-minutes 2
```

## GEX formula

Per-contract notional gamma exposure (Bn$ per 1% underlying move):

```
GEX = spot × gamma × open_interest × 100 × spot × 0.01 × sign
```

Calls contribute positive GEX; puts contribute negative GEX. Unusual Whales strike aggregates use trade-verified exposure and may differ from open-interest-only estimates.

## Forecasting

The prediction stack (`gex_core.predict`) combines:

- **Weighted KNN** on regime feature vectors with empirical prediction intervals
- **Calibrated confidence** from walk-forward sign accuracy
- **Market context** — realized vol, spot returns, 0DTE ratio (VIX when online)
- **Regime-conditional blending** with optional trained-model overlay (XGBoost / LSTM)
- **Structural attribution** — spot vs residual ΔGEX split (`gex_core.structural`)

Train models:

```bash
python scripts/train_gex_model.py --ticker SPX --lookback-days 90
python scripts/train_gex_lstm.py --ticker SPX --seq-len 8 --epochs 50
python scripts/backtest_gex_prediction.py --ticker SPX
```

## Live option flow

The `live/` package ingests JSON-lines flow events and computes real-time strike-level GEX deltas:

```bash
python live/ingest.py --feed data/flow_sample.jsonl --spot 4800
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `UW_API_KEY` | (required) | Unusual Whales API key |
| `GEX_REFRESH_INTERVAL_MINUTES` | `2` | Web dashboard auto-refresh (2-minute snapshots) |
| `GEX_BACKFILL_INTERVAL_MINUTES` | `2` | Sample UW 1-minute API rows every N minutes |
| `GEX_INTRADAY_BACKFILL_DAYS` | `90` | Default lookback for intraday backfill script |
| `GEX_DAILY_BACKFILL_DAYS` | `90` | EOD strike backfill via `gex_backfill_intraday.py` |
| `GEX_DASHBOARD_HISTORY_DAYS` | `90` | Recent snapshots for KNN / panels |
| `GEX_DASHBOARD_HISTORY_MAX` | `240` | Max strike CSVs loaded per dashboard request |
| `GEX_PREDICTION_LOOKBACK_DAYS` | `90` | KNN / forecast training window on dashboard |
| `GEX_TRAIN_LOOKBACK_DAYS` | `90` | Default lookback for `train_gex_model.py` |
| `GEX_DASHBOARD_TIMELINE_DAYS` | `90` | Spot vs levels timeline from backfill index |
| `GEX_SPX_PRICE_PERIOD` | `5d` | Yahoo Finance window for live SPX chart |
| `GEX_SPX_PRICE_INTERVAL` | `15m` | Yahoo Finance bar size for live SPX chart |
| `GEX_DISABLE_SCHEDULER` | off | Set `1` to disable background refresh |
| `GEX_DATA_FILTERS` | `1` | Set `0` to skip export data-quality filters |
| `GEX_MIN_OPEN_INTEREST` | `1` | Minimum OI per contract (export pipeline) |
| `GEX_MAX_STRIKE_DISTANCE_PCT` | `0.35` | Max strike distance from spot (export pipeline) |
| `GEX_FLOW_FEED` | `data/flow_sample.jsonl` | JSONL flow file for live overlay |
| `GEX_ALERT_WEBHOOK_URL` | off | Webhook URL for alert dispatch |
| `GEX_ALERT_AUTO_DISPATCH` | off | Auto-dispatch high-severity alerts |
| `GEX_INDEX_DB` | `data/gex_index.db` | SQLite export index path |

See [docs/DATA_QUALITY.md](docs/DATA_QUALITY.md), [docs/LIVE_FEED.md](docs/LIVE_FEED.md), and [config/spx.env.example](config/spx.env.example) for the full variable list (including gamma auto-trader toggles).

## Filters

Filters run at **four layers**: export data quality (before GEX is computed), gamma **signal** generation, **entry** approval (rule + AI advisor), and **exit** rules. The auto-trader reads SPX gamma snapshots and executes SPY 0DTE options by default (`GEX_SIGNAL_TICKER=SPX`, `GEX_EXECUTION_TICKER=SPY`).

```
UW option chain
      │
      ▼
[1] Data quality filters ──► strike GEX series (CSV/JSON exports)
      │
      ▼
[2] Signal filters (signals.py) ──► max-gamma candidate + trade strike
      │
      ▼
[3] Entry filters (filters.py) + advisor ──► approve / reject + confidence
      │
      ▼
[4] Position gates (engine / backtest) ──► open trade or skip
      │
      ▼
[5] Exit rules (exits.py) ──► stop, magnet touch, time stop, EOD flatten
```

Enable the trader with `GEX_AUTO_TRADER=1`. Backtest and tune filters:

```bash
python scripts/backtest_auto_trader.py --lookback-days 14 --starting-capital 500
python scripts/backtest_improvement_sweep.py --lookback-days 14
```

**Option marks (backtest):** By default PnL uses a synthetic leverage model (`GEX_TRADER_OPTION_LEVERAGE`). Set `GEX_TRADER_UW_OPTION_MARKS=1` with `UW_API_KEY` to mark SPY 0DTE entries/exits from UW intraday contract bars (`/api/option-contract/{symbol}/intraday`). Marks are cached under `data/uw_option_cache/`. Missing quotes fall back to the synthetic model.

### 1. Export data-quality filters

Applied in `gex_core.data_quality.clean_option_data()` when building each snapshot. These shape the strike distribution that all downstream gamma signals use — they are **not** the same as trader entry filters.

| Variable | Default | Effect |
|----------|---------|--------|
| `GEX_DATA_FILTERS` | `1` | Master switch (`0` = parse symbols only) |
| `GEX_MIN_OPEN_INTEREST` | `1` | Drop low-OI contracts |
| `GEX_MIN_GAMMA` | `0` | Drop non-positive gamma |
| `GEX_MAX_STRIKE_DISTANCE_PCT` | `0.35` | Drop far-OTM strikes |
| `GEX_MAX_IV` | `6.0` | Drop IV outliers |
| `GEX_MAX_BID_ASK_SPREAD_PCT` | `1.0` | Drop wide/crossed spreads |
| `GEX_DEDUPE_SYMBOLS` | `1` | Deduplicate option symbols |

Per-run removal counts are written to each `summary.json` under `data_quality`. See [docs/DATA_QUALITY.md](docs/DATA_QUALITY.md).

### 2. Signal-generation filters

Implemented in `gex_core.trading.signals.compute_entry_candidates()`. A snapshot must pass here before any entry filter runs.

| Check | Env / logic | Default | Skip reason |
|-------|-------------|---------|-------------|
| Max-gamma mode | `GEX_TRADER_MAX_GAMMA_ONLY=1` | on | Only the largest **positive** gamma magnet; no `fastest_gamma_increase` fallback |
| Gamma rising | compare current vs previous strike profile | — | `gamma_declined` if top magnet Δγ < 0 |
| Tradeable strike | `GEX_TRADER_MAX_STRIKE_DISTANCE_PCT` | `0.02` (2%) | `strike_too_far` if no ATM/slightly-ITM positive-γ strike in range |
| Strike selection | `GEX_TRADER_MAGNET_ANCHORED_STRIKES` | off | Off: nearest ATM positive-γ strike; on: trade at magnet strike |
| Multi-candidate | `GEX_TRADER_MULTI_STRIKE` | `2` | Up to N ranked magnets (legacy mode only; max-gamma-only emits one) |
| Direction lock | `master_direction` from magnet vs spot | — | Calls if magnet ≥ spot, puts if below (max-gamma-only) |
| Min gamma (legacy) | `GEX_TRADER_MIN_GAMMA_DELTA`, `GEX_TRADER_MIN_FASTEST_GAMMA_DELTA` | `0.03` | Used when `MAX_GAMMA_ONLY=0` for fastest-increase fallback |

### 3. Entry filters

`gex_core.trading.filters.evaluate_entry_filters()` runs inside the rule-based advisor (`advisor.py`) and the live AI advisor path. **`GEX_TRADER_CLEAR_FILTERS=0` (default)** keeps entry gates and advisor confidence floor active; set `GEX_TRADER_CLEAR_FILTERS=1` to bypass all gates, or `GEX_TRADER_STRICT_FILTERS=0` to bypass only the advisor filter block.

**Evaluation order** (first failure wins; `filter` key is set on reject):

| Order | Filter | Env | Default | Rule |
|-------|--------|-----|---------|------|
| 1 | Valid data | — | — | Spot and strike must be > 0 |
| 2 | Gamma delta | `GEX_TRADER_MIN_GAMMA_DELTA` | `0.03` | Recommended signal Δγ must meet floor |
| 3 | Direction lock | `master_direction` | — | Trade call/put must match max-gamma direction |
| 4 | Magnet distance | `GEX_TRADER_MIN_MAGNET_DISTANCE_PCT` | `0` | Reject if magnet too **close** to spot (optional) |
| 5 | Magnet progress | `GEX_TRADER_MIN_MAGNET_PROGRESS` | `0` | Spot must have closed ≥ N% of distance toward magnet since prior bar |
| 6 | Regime | `GEX_TRADER_REGIME_STRICT` | off | Block entries when regime contains `SHORT` |
| 7 | Entry time window | `GEX_TRADER_ENTRY_TIME_FILTER` | on | No entries before open+`ENTRY_AFTER_OPEN_MIN` or after close−`ENTRY_BEFORE_CLOSE_MIN` (ET) |
| 8 | Spot momentum | `GEX_TRADER_REQUIRE_MOMENTUM` | on | Spot must move in trade direction over `GEX_TRADER_MOMENTUM_BARS` (default 2) bars |
| 9 | Gamma flip side | `GEX_TRADER_REQUIRE_FLIP_SIDE` | on | Calls require spot > gamma flip; puts require spot < flip |
| 10 | Flow alignment | `GEX_TRADER_REQUIRE_FLOW_ALIGN` | **off** | When on: call needs net flow ≥ 0 and buy ratio ≥ `MIN_FLOW_BUY_RATIO`; put mirrored |

**Flow filter detail** (only when `GEX_TRADER_REQUIRE_FLOW_ALIGN=1`):

- `flow_buy_ratio` from snapshot: calls need ≥ `GEX_TRADER_MIN_FLOW_BUY_RATIO` (default `0.55`); puts need ≤ `1 − ratio`
- `flow_aggressiveness`: optional floor via `GEX_TRADER_MIN_FLOW_AGGRESSIVENESS` (default `0` = disabled)
- `flow_net_delta_gex_bn`: if |flow| ≥ 0.01, calls need flow ≥ 0, puts need flow ≤ 0; weak flow passes

**Market context** fields consumed from each snapshot (`MarketContext` in `filters.py`): `spot`, `gamma_flip`, `regime`, `flow_*`, `confluence_score`, `zero_dte_ratio`, `iv_rank`, `expected_move_pct`, CPI/NFP/FOMC flags, `spot_history`, `export_ts`.

### 4. Advisor and confidence gates

After entry filters, `advise_entry()` may approve or reject:

- **Rule-based path** (backtests): scores confidence from gamma score, Δγ, and journal win rate; applies `evaluate_entry_filters()` first
- **AI path** (live, when configured): LLM JSON verdict; filters re-checked on approve
- **Confidence floor**: `GEX_TRADER_MIN_ENTRY_CONFIDENCE` (default `0` = no floor) — reject if below
- **Strong setup**: `GEX_TRADER_STRONG_CONFIDENCE` + `GEX_TRADER_STRONG_GAMMA_DELTA` widen exits and contract count

### 5. Position and session gates

Applied in `engine.py` / `backtest.py` after advisor approval:

| Gate | Env | Default | Behavior |
|------|-----|---------|----------|
| Session | `GEX_TRADER_SESSION_ONLY` | on | Live ticks only during RTH (`GEX_TRADER_SESSION_*` hours) |
| Weekends | — | — | Backtest skips non-trading days |
| Max open | `GEX_TRADER_MAX_OPEN` | `2` | Cap concurrent positions |
| Entries per cycle | `GEX_TRADER_MAX_ENTRIES_PER_CYCLE` | `1` | Max new opens per gamma snapshot |
| Duplicate strike | — | — | Skip if same execution strike + option type already open |
| Stop cooldown | `GEX_TRADER_STOP_COOLDOWN_BARS` | `2` | No re-entry at same magnet after stop loss |
| Execution spot | `GEX_EXECUTION_TICKER` | `SPY` | Requires live SPY spot when signal is SPX |
| Risk sizing | `GEX_TRADER_RISK_SIZING` | on | Cap contracts by `GEX_TRADER_RISK_PER_TRADE_PCT` × equity |
| Capital | — | — | Skip if premium × 100 × qty exceeds cash |

### 6. Exit rules

`gex_core.trading.exits.evaluate_exit()` checks in order:

| Priority | Exit | Env | Default | Condition |
|----------|------|-----|---------|-----------|
| 1 | Stop loss | `GEX_TRADER_STOP_LOSS_PCT` | `20%` | PnL ≤ −stop; far-OTM uses `GEX_TRADER_FAR_OTM_STOP_PCT` (`3%`) beyond `FAR_OTM_DISTANCE_PCT` |
| 2 | Magnet touch | `GEX_TRADER_MAGNET_TOUCH_EXIT` | off | Opt-in; spot at magnet with min PnL (let `TAKE_PROFIT` run by default) |
| 3 | Take profit | `GEX_TRADER_TAKE_PROFIT_PCT` | `60%` | Full target; optional `GEX_TRADER_DYNAMIC_TP` scales down from IV |
| 4 | Magnet partial | `GEX_TRADER_MAGNET_PARTIAL_EXIT` | off | Exit fraction at `GEX_TRADER_MAGNET_PARTIAL_PROGRESS` (default 80%) toward magnet |
| 5 | Partial TP | `GEX_TRADER_PARTIAL_TP_PCT` | `8%` | Only when `hold_for_target=False` (non-strong profiles) |
| 6 | Trailing stop | `GEX_TRADER_TRAIL_TRIGGER_PCT` / `TRAIL_FLOOR_PCT` | `10%` / `5%` | After peak PnL hits trigger, exit at floor |
| 7 | Max hold | `GEX_TRADER_MAX_HOLD_MINUTES` | `30` min | Flat exit at max hold (≈15 bars at 2-min snapshots) if TP/SL not hit |
| 8 | Time stop | `GEX_TRADER_TIME_STOP_BARS` | max-hold bars | Stale losers: PnL < `TIME_STOP_MIN_PNL_PCT` and low magnet progress |
| 9 | EOD flatten | `GEX_TRADER_EOD_FLATTEN` | on | Close all at `EOD_FLATTEN_HOUR:MIN` (default 15:45 ET) |

**Exit profiles** (`build_exit_profile`): strong confidence + gamma, max-gamma-only, or near-magnet setups set `hold_for_target=True` (skip early partial TP, longer time stop, wider trail). `GEX_TRADER_DYNAMIC_TIME_STOP=1` extends time-stop bars when magnet is far from entry spot. `GEX_TRADER_FIX_MAGNET_EXIT_SCALE=1` maps SPX magnet levels to SPY scale for progress checks.

### 7. Trader env reference (filters & exits)

| Variable | Default | Layer |
|----------|---------|-------|
| `GEX_TRADER_CLEAR_FILTERS` | `0` | Set `1` to clear all entry filters |
| `GEX_TRADER_MIN_ENTRY_CONFIDENCE` | `0.55` | Minimum advisor confidence to open (0 when clear filters on) |
| `GEX_ADVISOR_CONTEXT_MAX_CHARS` | `24000` | UW + signal context cap for entry advisor LLM |
| `GEX_TRADER_STRICT_FILTERS` | `1` when clear off | Entry master switch |
| `GEX_TRADER_MAX_GAMMA_ONLY` | `1` | Signal |
| `GEX_TRADER_REQUIRE_MOMENTUM` | `1` | Entry |
| `GEX_TRADER_REQUIRE_FLIP_SIDE` | `1` | Entry |
| `GEX_TRADER_REQUIRE_FLOW_ALIGN` | `0` | Entry |
| `GEX_TRADER_ENTRY_TIME_FILTER` | `1` | Entry |
| `GEX_TRADER_ENTRY_AFTER_OPEN_MIN` | `15` | Entry window |
| `GEX_TRADER_ENTRY_BEFORE_CLOSE_MIN` | `30` | Entry window |
| `GEX_TRADER_MIN_GAMMA_DELTA` | `0.03` | Entry + signal |
| `GEX_TRADER_MAX_STRIKE_DISTANCE_PCT` | `0.02` | Signal strike pick |
| `GEX_TRADER_MAGNET_ANCHORED_STRIKES` | `0` | Signal strike pick |
| `GEX_TRADER_FIX_MAGNET_EXIT_SCALE` | `1` | Exit progress (SPX→SPY) |
| `GEX_TRADER_DYNAMIC_TIME_STOP` | `1` | Exit time stop |
| `GEX_TRADER_MAGNET_TOUCH_MIN_PNL_PCT` | `0.04` | Min gain before magnet-touch exit |
| `GEX_TRADER_CYCLE_SECONDS` | `15` | Live poll interval between snapshots |

Full trader block: [config/spx.env.example](config/spx.env.example).

### 8. Config present but not wired to entry filters

These exist in `gex_core.trading.config` but are **not** enforced in `evaluate_entry_filters()` today (reserved for Monte Carlo / future use):

- `GEX_TRADER_MIN_ZERO_DTE_RATIO`
- `GEX_TRADER_MAX_IV_RANK`
- `GEX_TRADER_MIN_CONFLUENCE`
- `GEX_TRADER_BLOCK_EVENTS` / `GEX_TRADER_EVENT_SIZE_MULT` (CPI/NFP/FOMC flags are stored on snapshots but not gated at entry)

Train ML overlays (optional):

```bash
pip install -r requirements-ml.txt
python scripts/train_gex_model.py --ticker SPX
```

## Health & ops

```bash
curl -s localhost:8080/health | jq
```

`uw_api_configured: false` in the `/health` payload (or a startup log warning
`UW_API_KEY is not set ...`) means the server has no API key. The dashboard then
shows **"Live data isn't configured on this server (UW_API_KEY is missing)"** and
serves only saved snapshots. Fix by setting `UW_API_KEY` in the **service**
environment:

- **docker compose:** export `UW_API_KEY` in your shell or add it to `.env` / `config/spx.env` before `docker compose up` (compose loads those files automatically).
- **Heroku/Procfile-style:** `heroku config:set UW_API_KEY=...` (or the platform's config UI).
- **systemd/bare gunicorn:** add `UW_API_KEY=...` to the unit's `Environment=`/`EnvironmentFile=`.
- **Cursor Cloud Agents:** add it under Dashboard → Cloud Agents → Secrets. The repo includes `.cursor/environment.json`, which starts the Flask dashboard via `scripts/start_web.sh` and inherits injected secrets.

Compact aged strike CSVs (keeps summaries + cumulative):

```bash
python scripts/gex_compact_exports.py --ticker SPX --keep-full-days 14
```

## Scheduled exports

- Daily: `.github/workflows/daily_exports.yml`
- Intraday (weekdays, every 2 minutes during US session): `.github/workflows/intraday_exports.yml`

See [ROADMAP.md](ROADMAP.md) for planned work.

## Documentation

- **[GitHub Wiki](https://github.com/TheMitchyBoy/GEX/wiki)** — full project guide (source: [`wiki/`](wiki/), auto-published on `main`)
- [Wiki one-time setup](docs/WIKI_SETUP.md) — required once before the publish Action can push
- [Getting started (wiki)](https://github.com/TheMitchyBoy/GEX/wiki/Getting-Started)
- [Roadmap (wiki)](https://github.com/TheMitchyBoy/GEX/wiki/Roadmap)
- [Improvement ideas backlog](docs/IMPROVEMENTS.md) — prioritized engineering tasks

## Project layout

| Path | Role |
|------|------|
| `main.py` | CLI entry — fetch, print, plot, export |
| `web_app.py` | Flask SPX dashboard |
| `streamlit_app.py` | Streamlit explorer |
| `gex_core/` | Core library (fetch, history, features, predict, `gex_core/trading/`) |
| `live/` | Real-time flow ingest |
| `scripts/` | Refresh, training, backtest, compaction |
| `gex_core/storage.py` | SQLite export index |
| `data/exports/` | Timestamped snapshot store |
| `models/` | Trained overlay models + manifests |

## Disclaimer

This tool is for educational and informational purposes only. GEX analysis reflects assumptions about dealer hedging and is not financial advice.
