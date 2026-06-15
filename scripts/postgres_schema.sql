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
    PRIMARY KEY (ticker, ts)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_ts
    ON snapshots (ticker, ts DESC);

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
