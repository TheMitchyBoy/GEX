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
    quality_score DOUBLE PRECISION,
    flip_confidence TEXT,
    regime_consistent BOOLEAN,
    spot_source TEXT,
    spot_disagreement_pct DOUBLE PRECISION,
    strike_profile_confidence TEXT,
    data_lag_sec DOUBLE PRECISION,
    uw_rate_limit_json JSONB,
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
    quality_score DOUBLE PRECISION,
    data_lag_sec DOUBLE PRECISION,
    uw_rate_limit_json JSONB,
    PRIMARY KEY (ticker, ts)
);

CREATE TABLE IF NOT EXISTS processor_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_quality_stats (
    ticker TEXT NOT NULL,
    market_date TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (ticker, market_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_quality_stats_ticker_date
    ON daily_quality_stats (ticker, market_date DESC);

CREATE TABLE IF NOT EXISTS prediction_accuracy_daily (
    ticker TEXT NOT NULL,
    market_date TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (ticker, market_date)
);

CREATE INDEX IF NOT EXISTS idx_prediction_accuracy_daily_ticker_date
    ON prediction_accuracy_daily (ticker, market_date DESC);

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

ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION;
ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS flip_confidence TEXT;
ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS regime_consistent BOOLEAN;
ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS spot_source TEXT;
ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS spot_disagreement_pct DOUBLE PRECISION;
ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS strike_profile_confidence TEXT;
ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS data_lag_sec DOUBLE PRECISION;
ALTER TABLE snapshot_features ADD COLUMN IF NOT EXISTS uw_rate_limit_json JSONB;
ALTER TABLE snapshot_diagnostics ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION;
ALTER TABLE snapshot_diagnostics ADD COLUMN IF NOT EXISTS data_lag_sec DOUBLE PRECISION;
ALTER TABLE snapshot_diagnostics ADD COLUMN IF NOT EXISTS uw_rate_limit_json JSONB;

CREATE MATERIALIZED VIEW IF NOT EXISTS latest_snapshot AS
SELECT DISTINCT ON (ticker)
    ticker, ts, market_date, spot, total_gex, regime, indexed_at, snapshot_at, prior_ts
FROM snapshots
ORDER BY ticker, ts DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_latest_snapshot_ticker ON latest_snapshot (ticker);

CREATE OR REPLACE VIEW training_snapshots AS
SELECT
    s.ticker,
    s.ts,
    s.market_date,
    s.spot,
    s.total_gex,
    s.regime,
    s.snapshot_at,
    f.quality_score,
    f.flip_confidence,
    f.regime_consistent,
    f.strike_count,
    f.delta_gex,
    f.spot_return,
    d.status AS diagnostic_status
FROM snapshots s
JOIN snapshot_features f ON f.ticker = s.ticker AND f.ts = s.ts
LEFT JOIN snapshot_diagnostics d ON d.ticker = s.ticker AND d.ts = s.ts
WHERE COALESCE(d.status, 'ok') IN ('ok', 'ok_with_warnings')
  AND COALESCE(f.quality_score, 0) >= 0.8
  AND COALESCE(f.strike_profile_confidence, 'high') <> 'low'
  AND COALESCE(s.summary_json->>'strike_profile_source', '') <> 'eod_scaled';
