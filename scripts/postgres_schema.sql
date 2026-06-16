-- GEX application schema for Railway PostgreSQL
-- Safe to run multiple times (CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS)
-- Or run: python3 scripts/init_postgres_schema.py

CREATE TABLE IF NOT EXISTS snapshots (
    ticker TEXT NOT NULL,
    ts TEXT NOT NULL,
    market_date TEXT,
    spot DOUBLE PRECISION,
    total_gex DOUBLE PRECISION,
    regime TEXT,
    summary_path TEXT,
    strike_path TEXT,
    indexed_at TEXT,
    summary_json JSONB,
    expiration_json JSONB,
    surface_json JSONB,
    greek_exposure_json JSONB,
    PRIMARY KEY (ticker, ts)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_ts
    ON snapshots (ticker, ts DESC);

CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_date
    ON snapshots (ticker, market_date);

CREATE TABLE IF NOT EXISTS snapshot_strikes (
    ticker TEXT NOT NULL,
    ts TEXT NOT NULL,
    strike DOUBLE PRECISION NOT NULL,
    gex_bn_per_pct DOUBLE PRECISION,
    cumulative_gex_bn_per_pct DOUBLE PRECISION,
    PRIMARY KEY (ticker, ts, strike)
);

CREATE INDEX IF NOT EXISTS idx_snapshot_strikes_ticker_ts
    ON snapshot_strikes (ticker, ts);

CREATE TABLE IF NOT EXISTS snapshot_strikes_atm (
    ticker TEXT NOT NULL,
    ts TEXT NOT NULL,
    strike DOUBLE PRECISION NOT NULL,
    gex_bn_per_pct DOUBLE PRECISION,
    cumulative_gex_bn_per_pct DOUBLE PRECISION,
    PRIMARY KEY (ticker, ts, strike)
);

CREATE INDEX IF NOT EXISTS idx_snapshot_strikes_atm_ticker_ts
    ON snapshot_strikes_atm (ticker, ts);

CREATE TABLE IF NOT EXISTS snapshot_features (
    ticker TEXT NOT NULL,
    ts TEXT NOT NULL,
    prior_ts TEXT,
    snapshot_at TIMESTAMPTZ,
    gamma_flip DOUBLE PRECISION,
    call_wall DOUBLE PRECISION,
    put_wall DOUBLE PRECISION,
    pos_gamma_peak_strike DOUBLE PRECISION,
    flip_distance_pct DOUBLE PRECISION,
    wall_spread DOUBLE PRECISION,
    gex_concentration DOUBLE PRECISION,
    near_term_ratio DOUBLE PRECISION,
    zero_dte_ratio DOUBLE PRECISION,
    term_curvature DOUBLE PRECISION,
    expiration_count DOUBLE PRECISION,
    front_term_ratio DOUBLE PRECISION,
    back_term_ratio DOUBLE PRECISION,
    delta_gex DOUBLE PRECISION,
    delta_spot DOUBLE PRECISION,
    spot_return DOUBLE PRECISION,
    regime_changed BOOLEAN,
    surface_vector JSONB,
    strike_profile_hash TEXT,
    strike_count INTEGER,
    PRIMARY KEY (ticker, ts)
);

CREATE INDEX IF NOT EXISTS idx_snapshot_features_ticker_ts
    ON snapshot_features (ticker, ts DESC);

CREATE TABLE IF NOT EXISTS snapshot_diagnostics (
    ticker TEXT NOT NULL,
    ts TEXT NOT NULL,
    status TEXT NOT NULL,
    validation_json JSONB,
    uw_fetch_ms DOUBLE PRECISION,
    postgres_write_ms DOUBLE PRECISION,
    indexed_at TEXT,
    PRIMARY KEY (ticker, ts)
);

CREATE TABLE IF NOT EXISTS processor_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    option_type TEXT NOT NULL,
    strike DOUBLE PRECISION NOT NULL,
    qty DOUBLE PRECISION NOT NULL DEFAULT 1,
    entry_ts TEXT NOT NULL,
    exit_ts TEXT,
    entry_spot DOUBLE PRECISION NOT NULL,
    exit_spot DOUBLE PRECISION,
    entry_premium DOUBLE PRECISION NOT NULL,
    exit_premium DOUBLE PRECISION,
    pnl_pct DOUBLE PRECISION,
    pnl_usd DOUBLE PRECISION,
    exit_reason TEXT,
    signal_type TEXT,
    signal_strike DOUBLE PRECISION,
    signal_gamma DOUBLE PRECISION,
    gamma_delta DOUBLE PRECISION,
    ai_confidence DOUBLE PRECISION,
    ai_reason TEXT,
    meta_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status, entry_ts DESC);

CREATE TABLE IF NOT EXISTS decisions (
    id SERIAL PRIMARY KEY,
    ts TEXT NOT NULL,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,
    payload_json TEXT,
    ai_verdict TEXT,
    ai_notes TEXT
);

CREATE TABLE IF NOT EXISTS trader_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_predictions (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    source TEXT NOT NULL,
    snapshot_ts TEXT,
    market_date TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    payload_json TEXT NOT NULL,
    actual_json TEXT,
    outcome_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_llm_predictions_open
    ON llm_predictions (ticker, resolved_at, created_at DESC);

CREATE TABLE IF NOT EXISTS daily_insights (
    ticker TEXT NOT NULL,
    market_date TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (ticker, market_date, kind)
);

CREATE INDEX IF NOT EXISTS idx_daily_insights_ticker_date
    ON daily_insights (ticker, market_date DESC);

-- Migrations for databases created before JSONB columns existed
ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS summary_json JSONB;
ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS expiration_json JSONB;
ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS surface_json JSONB;
ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS greek_exposure_json JSONB;
ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS snapshot_at TIMESTAMPTZ;
ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS prior_ts TEXT;
CREATE INDEX IF NOT EXISTS idx_snapshots_snapshot_at ON snapshots (ticker, snapshot_at DESC);

CREATE MATERIALIZED VIEW IF NOT EXISTS latest_snapshot AS
SELECT DISTINCT ON (ticker)
    ticker, ts, market_date, spot, total_gex, regime, indexed_at, snapshot_at, prior_ts
FROM snapshots
ORDER BY ticker, ts DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_latest_snapshot_ticker ON latest_snapshot (ticker);
