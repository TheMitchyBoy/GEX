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
       └─ web_app.py (Flask + gunicorn)     ├─ gex_core.predict  (KNN forecast)
              │                             └─ scripts/train_*   (model overlay)
              ├─ Background scheduler (UW refresh, optional auto-traders)
              ├─ UW price websocket (live spot chart)
              └─ SQLite journals (gex_index.db, trading_journal.db)
```

Persistence is **file-based** with an optional **SQLite index** (`data/gex_index.db`) for fast history lookups. Each refresh writes a matched set of files sharing a timestamp suffix under `data/exports/`.

## App setup

The primary interface is the **Flask web dashboard** (`web_app.py`), served by **gunicorn** in production via `scripts/start_web.sh` / `wsgi.py`. On boot the app:

1. Loads configuration from `.env` and `config/spx.env` (`gex_core.env_bootstrap`)
2. Starts a background scheduler (unless `GEX_DISABLE_SCHEDULER=1`) that pulls UW data on an interval
3. Optionally runs **two independent auto-traders** on separate timers (Wall GEX and Gamma Magnet)
4. Serves dashboards from saved exports, with live UW websocket prices when configured

### Prerequisites

| Requirement | Required for |
|-------------|--------------|
| Python 3.11+ | All local runs |
| `UW_API_KEY` | Live data, refresh, dashboards |
| Webull OpenAPI credentials | Live SPY 0DTE orders (`/trade`) |
| `OPENAI_API_KEY` or `OPENROUTER_API_KEY` | AI entry advisor and chat (optional) |

### 1. Install

```bash
git clone https://github.com/TheMitchyBoy/GEX.git
cd GEX
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

### 2. Configure environment

Copy the example config and set at least your Unusual Whales key:

```bash
cp config/spx.env.example config/spx.env
# edit config/spx.env:
export UW_API_KEY=your-key-here
```

**How env loading works:** `bootstrap_env()` reads `.env` and `config/spx.env` in order. Variables already set in the process environment (shell export, Docker, Railway, Cursor Cloud secrets) are **never overwritten** by files.

| Variable | Default | Purpose |
|----------|---------|---------|
| `UW_API_KEY` | (none) | **Required** — Unusual Whales API access |
| `GEX_DATA_DIR` | `data/` | Exports, SQLite DBs; use `/app/data` on Railway with a volume |
| `GEX_DEFAULT_TICKERS` | `SPX` | Symbols refreshed by the scheduler |
| `GEX_REFRESH_INTERVAL_MINUTES` | `2` | How often new GEX snapshots are fetched (minimum 60s in scheduler) |
| `GEX_DISABLE_SCHEDULER` | off | Set `1` to disable background refresh |

Full variable list: [config/spx.env.example](config/spx.env.example).

### 3. Bootstrap historical data

A fresh install has no strike history. Forecasts, timelines, and backtests need CSV snapshots under `data/exports/`.

**Automatic (default):** when fewer than 4 strike CSVs exist, `scripts/start_web.sh` kicks off a background 90-day backfill (`GEX_AUTO_BACKFILL_IF_EMPTY=1`).

**Manual / first deploy:**

```bash
export GEX_STARTUP_BACKFILL=1          # one-shot on container boot
# or run directly:
python scripts/gex_backfill_intraday.py --tickers SPX --intraday-days 90 --interval-minutes 2
python scripts/gex_refresh.py --force
```

GitHub Actions also run scheduled exports (see [Scheduled exports](#scheduled-exports) below).

### 4. Run locally

**Development** — Flask built-in server on port 5000:

```bash
export UW_API_KEY=your-key
python web_app.py
# → http://localhost:5000/
```

**Production-like** — gunicorn on port 8080 (same path used in Docker/Railway):

```bash
bash scripts/start_web.sh
# → http://localhost:8080/
```

Verify the server is healthy:

```bash
curl -s localhost:8080/health | jq
```

`uw_api_configured: false` means no `UW_API_KEY` — the UI will show saved snapshots only.

### 5. Docker

```bash
cp config/spx.env.example config/spx.env   # set UW_API_KEY
docker compose up web
```

Open **http://localhost:8080/**. The `data/` directory is bind-mounted so exports and journals survive restarts.

One-off refresh (tools profile):

```bash
docker compose --profile tools run --rm refresh
```

### 6. Deploy to Railway / cloud

1. Set `UW_API_KEY` in the service environment variables.
2. Mount a **persistent volume** at `/app/data` and set `GEX_DATA_DIR=/app/data`.
3. Start command: `bash scripts/start_web.sh` (default in the Dockerfile).
4. Optional first deploy: `GEX_STARTUP_BACKFILL=1` to pull 90 days of intraday history.
5. `GEX_BACKTEST_METRICS=0` (default) disables walk-forward backtest history loading on the web app. `GEX_DASHBOARD_SKIP_BACKTEST=1` skips backtest panels on dashboard API payloads.
6. `GEX_DAILY_LEARNING=0` (default) skips the startup lesson cycle and returns a fast disabled response from `/api/agent/daily-strategy`; the Periscope UI loads strategy only when you click **Load today's strategy**. Set `GEX_DAILY_LEARNING=1` to re-enable.
7. `GEX_PAGE_MINIMAL_LOAD=1` (default) loads **only the current gamma snapshot** on dashboard pages — no historical replay catalog or prior-slice trails. Scheduled refresh still writes every export to `data/exports/` for backfill, training, and backtests. When minimal load is on, `/api/agent/daily-strategy` also skips `_prediction_history` (240 snapshots).
8. `GEX_PAGE_UW_PEEK_ONLY=1` (default) avoids blocking HTML on live UW HTTP — pages paint from exports/cache immediately; the browser fetches fresh UW data via `?live=1` on strategy/status refresh (4s timeout via `GEX_UW_FETCH_TIMEOUT_SEC`).
9. After the first backfill, consider `GEX_DASHBOARD_HISTORY_DAYS=30`, `GEX_AUTO_BACKFILL_IF_EMPTY=0`, and `GEX_STARTUP_BACKFILL=0` to reduce disk I/O on deploy.
10. Railway health check uses `/health/live` (instant). If the site shows `ERR_CONNECTION_ABORTED`, open **Deployments → Logs** — common causes are OOM during model retrain (`GEX_RETRAIN_ON_START=1`), missing volume at `/app/data`, or a crashed gunicorn worker. Set `GEX_DATA_DIR=/app/data` and mount a persistent volume.

### 7. Pages and auto-traders

| URL | Dashboard | Background trader | Enable with |
|-----|-----------|-------------------|-------------|
| `/` | **Wall GEX** — trade toward the lowest-\|γ\| wall | `low_gex_engine.py` | `GEX_WALL_GEX_AUTO=1` (default on) |
| `/gamma`, `/periscope` | **Gamma Magnet** — max-positive-γ magnet strategy | `engine.py` | `GEX_AUTO_TRADER=1` |
| `/gamma/near`, `/near` | **Near-Spot Walls** — low/high γ walls within ±1% of spot | `low_gex_engine.py` | `GEX_WALL_GEX_AUTO=1` |
| `/trade` | **Webull quick-trade desk** — live quotes + one-click orders | manual / API | Webull credentials |
| `/ticker/<TICKER>/...` | Per-ticker variants of the above | same | same |

Both auto-traders read **SPX** gamma signals (`GEX_SIGNAL_TICKER=SPX`) and execute **SPY** 0DTE options by default (`GEX_EXECUTION_TICKER=SPY`). Default exits are **3% stop / 22% take profit** on full-window Wall GEX (`/`); **near-spot walls** (`/near`, ±1%) use **3% / 28%** with **10-bar max hold** and **wall-shift re-entry off** (`GEX_NEAR_WALL_*`). Gamma Magnet uses **20% / 60%** unless overridden via `GEX_TRADER_STOP_LOSS_PCT` / `GEX_TRADER_TAKE_PROFIT_PCT`.

**Scheduler loops** (when not disabled):

| Job | Interval | Env |
|-----|----------|-----|
| UW data refresh | `GEX_REFRESH_INTERVAL_MINUTES` (2 min) | always (unless scheduler off) |
| Gamma Magnet tick | `GEX_TRADER_CYCLE_SECONDS` (15 s) | `GEX_AUTO_TRADER=1` |
| Wall GEX tick | `GEX_WALL_GEX_CYCLE_SECONDS` (30 s) | `GEX_WALL_GEX_AUTO=1` |

Arm or disarm traders from the dashboard UI, or via `POST /api/trader/arm` and `POST /api/wall-gex/arm`. Paper mode is the default (`GEX_TRADER_PAPER=1`).

**Streamlit explorer** (optional, separate from the main dashboard):

```bash
streamlit run streamlit_app.py
```

### 8. Optional: live Webull trading

```bash
export GEX_TRADER_PAPER=0
export GEX_WEBULL_APP_KEY=your-app-key
export GEX_WEBULL_APP_SECRET=your-app-secret
export GEX_WEBULL_ACCOUNT_ID=your-account-id
export GEX_TRADER_LIVE_CONFIRM=1
export GEX_ADMIN_TOKEN=change-me    # required for POST order routes when set
```

Orders from `/trade` require `live_confirm: true` in the API body. Enable background execution with `GEX_AUTO_TRADER=1` and/or `GEX_WALL_GEX_AUTO=1`, then arm the trader on the dashboard.

### 9. CLI and manual refresh

**CLI** — fetch, plot, and export without the web server:

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

**Manual refresh scripts:**

```bash
python scripts/gex_refresh.py --force
python scripts/gex_refresh.py --tickers SPX --backfill-days 7
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
| `GEX_PAGE_UW_PEEK_ONLY` | `1` | HTML/API handlers use cached UW only; JS refresh passes `live=1` |
| `GEX_UW_FETCH_TIMEOUT_SEC` | `4` | Max seconds to block on live UW fetch when `live=1` |
| `GEX_PAGE_MINIMAL_LOAD` | `1` | Current slice only on pages (no historical replay); exports still saved |
| `GEX_DAILY_LEARNING` | `0` | Skip startup lesson cycle; daily-strategy API returns disabled unless `1` |
| `GEX_AGENT_FETCH_EXTRAS` | `0` | Skip extra UW API calls per chat message |
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
python scripts/backtest_low_gex_trader.py --lookback-days 14 --window-pct 0.01
python scripts/compare_wall_gex_backtest.py --lookback-days 14 --window-pct 0.01
```

The compare script runs min vs max γ wall on the same history; for `/near` (±1%) it auto-applies the near-wall profile and keeps all snapshots (`dedupe` off). Use `--dedupe` only for full-window bar-collapse experiments.

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
| Max-gamma mode | `GEX_TRADER_MAX_GAMMA_ONLY=1` | on | Trade highest **+γ** magnet near spot (no `fastest_gamma_increase` fallback) |
| Negative γ walls | `GEX_TRADER_TRADE_NEGATIVE_GAMMA` | off | Set `1` to also enter on lowest **−γ** when \|γ\| dominates |
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
| 1 | Stop loss | `GEX_TRADER_STOP_LOSS_PCT` | `6%` | PnL ≤ −stop; far-OTM uses `GEX_TRADER_FAR_OTM_STOP_PCT` (`3%`) beyond `FAR_OTM_DISTANCE_PCT` |
| 2 | Magnet touch | `GEX_TRADER_MAGNET_TOUCH_EXIT` | off | Opt-in; spot at magnet with min PnL (let `TAKE_PROFIT` run by default) |
| 3 | Take profit | `GEX_TRADER_TAKE_PROFIT_PCT` | `28%` | Full target; optional `GEX_TRADER_DYNAMIC_TP` scales down from IV |
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
| `GEX_TRADER_MIN_ENTRY_CONFIDENCE` | `0` | Minimum advisor confidence to open (0 when clear filters on) |
| `GEX_ADVISOR_CONTEXT_MAX_CHARS` | `24000` | UW + signal context cap for entry advisor LLM |
| `GEX_TRADER_STRICT_FILTERS` | `0` when clear off | Entry master switch |
| `GEX_TRADER_MAX_GAMMA_ONLY` | `1` | Signal |
| `GEX_TRADER_TRADE_NEGATIVE_GAMMA` | `0` | Signal — lowest −γ walls |
| `GEX_TRADER_REQUIRE_MOMENTUM` | `0` | Entry |
| `GEX_TRADER_REQUIRE_FLIP_SIDE` | `0` | Entry |
| `GEX_TRADER_REQUIRE_FLOW_ALIGN` | `0` | Entry |
| `GEX_TRADER_ENTRY_TIME_FILTER` | `0` | Entry |
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
curl -s localhost:8080/health/ready | jq
```

If `uw_api_configured` is `false`, set `UW_API_KEY` in the service environment (see [App setup](#app-setup)). The dashboard still serves saved snapshots from `data/exports/`.

| Platform | Where to set `UW_API_KEY` |
|----------|---------------------------|
| Local / Docker | `.env`, `config/spx.env`, or shell `export` |
| Railway / Heroku | Service environment variables in the platform UI |
| systemd | `Environment=` / `EnvironmentFile=` in the unit |
| Cursor Cloud Agents | Dashboard → Cloud Agents → Secrets |

### Webull 401 `INVALID_TOKEN`

The trade desk banner means Webull rejected the stored OAuth token. The app clears the stale file automatically and pauses live API calls for ~30 seconds.

**Checklist:**

1. **Credentials** — `GEX_WEBULL_APP_KEY`, `GEX_WEBULL_APP_SECRET`, and `GEX_WEBULL_ACCOUNT_ID` must match the same approved OpenAPI app. Production uses `api.webull.com` (default). Set `GEX_WEBULL_USE_UAT=1` only with sandbox keys.
2. **Persistent token** — the token is stored at `$GEX_DATA_DIR/webull/token.txt` (default `data/webull/token.txt`). On Railway, mount a volume at `/app/data` so the token survives redeploys; otherwise you must re-approve in the mobile app after every restart.
3. **Mobile verification** — on first connect or after a token reset, open the Webull mobile app and approve the API verification prompt.
4. **Retry** — disarm the trader, wait ~30 seconds, then click **Retry connection** on `/trade` or the Wall GEX page. If `GEX_ADMIN_TOKEN` is set, enter it in the trade desk admin field first.
5. **Logs** — look for `init_token` / `check_token` lines from the Webull SDK; `status=PENDING` means approval is still waiting.

### Webull 401 `subscribe to US_OPTION quotes`

If the last error says **Insufficient permission, please subscribe to US_OPTION quotes**, auth is working but your OpenAPI app lacks the **US options quote** market-data entitlement. This is not a stale token — retrying connection will not help.

1. Open the [Webull OpenAPI developer portal](https://developer.webull.com/) and select your **production** app (the one matching `GEX_WEBULL_APP_KEY`).
2. Subscribe to **US options quotes** (`US_OPTION` / option market data). Wait for approval if the portal shows a pending state.
3. Confirm the linked brokerage account has options permissions and any required Webull data subscriptions.
4. Redeploy is not required; refresh `/trade` after the entitlement is active. Live bid/ask should populate; until then the desk uses estimated marks.

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
| `web_app.py` | Flask dashboard — routes, scheduler, trader hooks |
| `wsgi.py` | Gunicorn entrypoint (`bootstrap_env` + `APP`) |
| `scripts/start_web.sh` | Production start — backfill, retrain, gunicorn |
| `streamlit_app.py` | Streamlit explorer (optional) |
| `gex_core/` | Core library (fetch, history, features, predict, `gex_core/trading/`) |
| `gex_core/trading/` | Auto-traders: `engine.py` (Gamma Magnet), `low_gex_engine.py` (Wall GEX) |
| `config/spx.env.example` | Documented environment template |
| `live/` | Real-time flow ingest |
| `scripts/` | Refresh, training, backtest, compaction |
| `data/exports/` | Timestamped GEX snapshot store |
| `data/gex_index.db` | SQLite export index |
| `data/trading_journal.db` | Paper/live trade journal |
| `models/` | Trained overlay models + manifests |

## Disclaimer

This tool is for educational and informational purposes only. GEX analysis reflects assumptions about dealer hedging and is not financial advice.
