-- ============================================================================
-- Trading Backtester V2 — Complete Database Schema
-- TimescaleDB 2.x on PostgreSQL 15+
-- ============================================================================
-- Run: psql -U ava -d ava -f schema.sql
-- Requires: CREATE EXTENSION IF NOT EXISTS timescaledb;
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS backtester;
SET search_path TO backtester, public;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================================
-- 1. CANDLES — Daily OHLCV price data
-- ============================================================================
-- Primary time-series table for end-of-day prices. One row per symbol per day.
-- Compressed after 7 days; partitioned by time + symbol for fast range scans.

CREATE TABLE backtester.candles_daily (
    symbol        TEXT           NOT NULL,
    ts            TIMESTAMPTZ    NOT NULL,  -- market close timestamp (date at 16:00 ET)
    open          NUMERIC(12,4)  NOT NULL,
    high          NUMERIC(12,4)  NOT NULL,
    low           NUMERIC(12,4)  NOT NULL,
    close         NUMERIC(12,4)  NOT NULL,
    volume        BIGINT         NOT NULL,
    adj_close     NUMERIC(12,4),
    split_factor  NUMERIC(8,4)   DEFAULT 1.0,
    dividend      NUMERIC(8,4)   DEFAULT 0.0,
    data_source   TEXT           NOT NULL DEFAULT 'fmp',
    quality_score NUMERIC(4,3)   DEFAULT 1.000,
    created_at    TIMESTAMPTZ    DEFAULT NOW(),

    PRIMARY KEY (symbol, ts)
);

SELECT create_hypertable('backtester.candles_daily', 'ts');
SELECT add_dimension('backtester.candles_daily', 'symbol', number_partitions => 4);

ALTER TABLE backtester.candles_daily SET (
    timescaledb.compress,
    timescaledb.compress_segmentby  = 'symbol',
    timescaledb.compress_orderby    = 'ts DESC'
);
SELECT add_compression_policy('backtester.candles_daily', INTERVAL '7 days');

CREATE INDEX ix_candles_daily_sym_ts ON backtester.candles_daily (symbol, ts DESC);

-- ============================================================================
-- 2. CANDLES — Intraday OHLCV price data
-- ============================================================================
-- Stores 1m / 5m / 15m / 1h bars. Timeframe column distinguishes granularity.

CREATE TABLE backtester.candles_intraday (
    symbol        TEXT           NOT NULL,
    ts            TIMESTAMPTZ    NOT NULL,
    timeframe     TEXT           NOT NULL,  -- '1m','5m','15m','1h'
    open          NUMERIC(12,4)  NOT NULL,
    high          NUMERIC(12,4)  NOT NULL,
    low           NUMERIC(12,4)  NOT NULL,
    close         NUMERIC(12,4)  NOT NULL,
    volume        BIGINT         NOT NULL,
    data_source   TEXT           NOT NULL DEFAULT 'fmp',
    quality_score NUMERIC(4,3)   DEFAULT 1.000,
    created_at    TIMESTAMPTZ    DEFAULT NOW(),

    PRIMARY KEY (symbol, timeframe, ts)
);

SELECT create_hypertable('backtester.candles_intraday', 'ts');
SELECT add_dimension('backtester.candles_intraday', 'symbol', number_partitions => 8);

ALTER TABLE backtester.candles_intraday SET (
    timescaledb.compress,
    timescaledb.compress_segmentby  = 'symbol,timeframe',
    timescaledb.compress_orderby    = 'ts DESC'
);
SELECT add_compression_policy('backtester.candles_intraday', INTERVAL '3 days');

CREATE INDEX ix_candles_intra_sym_tf_ts
    ON backtester.candles_intraday (symbol, timeframe, ts DESC);

-- ============================================================================
-- 3. STRATEGIES — Strategy registry with versioning
-- ============================================================================
-- Each row is a unique (name, version) pair. Stores code hash + default config.

CREATE TABLE backtester.strategies (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT           NOT NULL,
    version         TEXT           NOT NULL,  -- semver, e.g. '2.1.0'
    description     TEXT,
    code_hash       TEXT           NOT NULL,  -- sha256 of strategy source
    default_config  JSONB          NOT NULL DEFAULT '{}',
    tags            TEXT[]         DEFAULT '{}',
    author          TEXT,
    changelog       TEXT,
    is_active       BOOLEAN        DEFAULT TRUE,
    created_at      TIMESTAMPTZ    DEFAULT NOW(),
    updated_at      TIMESTAMPTZ    DEFAULT NOW(),

    UNIQUE (name, version)
);

CREATE INDEX ix_strategies_name ON backtester.strategies (name);

-- ============================================================================
-- 4. BACKTESTS — Backtest run metadata
-- ============================================================================
-- One row per backtest execution. Links to a strategy and stores run config.

CREATE TABLE backtester.backtests (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id      UUID           NOT NULL REFERENCES backtester.strategies(id),
    strategy_name    TEXT           NOT NULL,
    strategy_version TEXT           NOT NULL,
    param_yaml       TEXT           NOT NULL,
    param_hash       TEXT           NOT NULL,
    run_type         TEXT           NOT NULL DEFAULT 'backtest',
        -- 'backtest', 'walk_forward', 'monte_carlo', 'optimization'
    parent_id        UUID           REFERENCES backtester.backtests(id),
    universe_name    TEXT           NOT NULL,
    start_date       DATE           NOT NULL,
    end_date         DATE           NOT NULL,
    initial_capital  NUMERIC(15,2)  NOT NULL,
    status           TEXT           NOT NULL DEFAULT 'pending',
        -- 'pending','running','completed','failed','cancelled'
    progress_pct     NUMERIC(5,2)   DEFAULT 0,
    symbols_total    INTEGER,
    symbols_done     INTEGER        DEFAULT 0,
    error_message    TEXT,
    metadata         JSONB          DEFAULT '{}',
    created_at       TIMESTAMPTZ    DEFAULT NOW(),
    started_at       TIMESTAMPTZ,
    completed_at     TIMESTAMPTZ,
    duration_seconds NUMERIC(10,3)
);

CREATE INDEX ix_backtests_strategy   ON backtester.backtests (strategy_name, created_at DESC);
CREATE INDEX ix_backtests_status     ON backtester.backtests (status);
CREATE INDEX ix_backtests_created    ON backtester.backtests (created_at DESC);

-- ============================================================================
-- 5. BACKTEST_RESULTS — Aggregate performance metrics per backtest
-- ============================================================================
-- Stores computed metrics (Sharpe, CAGR, drawdown, etc.) for fast querying.

CREATE TABLE backtester.backtest_results (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_id         UUID NOT NULL UNIQUE REFERENCES backtester.backtests(id) ON DELETE CASCADE,
    total_return_pct    NUMERIC(10,4),
    cagr_pct            NUMERIC(10,4),
    sharpe_ratio        NUMERIC(8,4),
    sortino_ratio       NUMERIC(8,4),
    calmar_ratio        NUMERIC(8,4),
    max_drawdown_pct    NUMERIC(8,4),
    max_drawdown_days   INTEGER,
    win_rate_pct        NUMERIC(6,2),
    profit_factor       NUMERIC(8,4),
    avg_trade_pnl       NUMERIC(15,2),
    avg_winner          NUMERIC(15,2),
    avg_loser           NUMERIC(15,2),
    best_trade_pnl      NUMERIC(15,2),
    worst_trade_pnl     NUMERIC(15,2),
    total_trades        INTEGER,
    winning_trades      INTEGER,
    losing_trades       INTEGER,
    avg_hold_days       NUMERIC(8,2),
    exposure_pct        NUMERIC(6,2),
    final_equity        NUMERIC(15,2),
    -- Equity curve stored as JSONB array of {date, equity} for charting
    equity_curve        JSONB,
    monthly_returns     JSONB,       -- {year-month: pct}
    metrics_extra       JSONB DEFAULT '{}',  -- extensible bucket
    computed_at         TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- 6. TRADES — Individual trade records
-- ============================================================================
-- One row per round-trip trade. Hypertable on entry_date for fast time-range queries.

CREATE TABLE backtester.trades (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_id             UUID           NOT NULL REFERENCES backtester.backtests(id) ON DELETE CASCADE,
    symbol                  TEXT           NOT NULL,
    direction               TEXT           NOT NULL CHECK (direction IN ('long','short')),
    entry_date              DATE           NOT NULL,
    entry_time              TIME,
    entry_price             NUMERIC(12,4)  NOT NULL,
    exit_date               DATE,
    exit_time               TIME,
    exit_price              NUMERIC(12,4),
    shares                  INTEGER        NOT NULL,
    pnl                     NUMERIC(15,2),
    pnl_pct                 NUMERIC(8,4),
    commission_cost         NUMERIC(10,2)  DEFAULT 0,
    slippage_cost           NUMERIC(10,2)  DEFAULT 0,
    entry_signal            JSONB,
    exit_reason             TEXT,
    hold_days               INTEGER,
    max_favorable_excursion NUMERIC(15,2),
    max_adverse_excursion   NUMERIC(15,2),
    regime_at_entry         TEXT,
    position_size_pct       NUMERIC(6,3),
    metadata                JSONB          DEFAULT '{}'
);

SELECT create_hypertable('backtester.trades', 'entry_date');

CREATE INDEX ix_trades_backtest ON backtester.trades (backtest_id, entry_date);
CREATE INDEX ix_trades_symbol   ON backtester.trades (symbol, entry_date DESC);

-- ============================================================================
-- 7. SIGNALS — Raw signal log (optional, can be large)
-- ============================================================================
-- Stores every signal generated by a strategy during a backtest.
-- Useful for debugging and signal analysis. Hypertable on ts.

CREATE TABLE backtester.signals (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_id  UUID          NOT NULL REFERENCES backtester.backtests(id) ON DELETE CASCADE,
    symbol       TEXT          NOT NULL,
    ts           TIMESTAMPTZ   NOT NULL,
    action       TEXT          NOT NULL,  -- 'buy','sell','hold','exit'
    strength     NUMERIC(4,3),
    confidence   NUMERIC(4,3),
    indicators   JSONB,        -- snapshot of indicator values at signal time
    metadata     JSONB         DEFAULT '{}'
);

SELECT create_hypertable('backtester.signals', 'ts');

CREATE INDEX ix_signals_backtest_ts ON backtester.signals (backtest_id, ts);

-- ============================================================================
-- 8. REGIME_HISTORY — Historical market regime classifications
-- ============================================================================
-- Time-series of regime labels (bull/bear/neutral/crisis) produced by the
-- regime classifier. Used for regime-filtered backtests.

CREATE TABLE backtester.regime_history (
    ts              TIMESTAMPTZ   NOT NULL,
    regime          TEXT          NOT NULL,  -- 'bull','bear','neutral','crisis'
    confidence      NUMERIC(4,3) NOT NULL,
    classifier      TEXT          NOT NULL DEFAULT 'default',
    features        JSONB,
    PRIMARY KEY (classifier, ts)
);

SELECT create_hypertable('backtester.regime_history', 'ts');

CREATE INDEX ix_regime_hist_ts ON backtester.regime_history (ts DESC);

-- ============================================================================
-- 9. REGIME_STATE — Current / latest regime snapshot
-- ============================================================================
-- Convenience table holding the most recent regime for each classifier.

CREATE TABLE backtester.regime_state (
    classifier   TEXT PRIMARY KEY,
    regime       TEXT          NOT NULL,
    confidence   NUMERIC(4,3) NOT NULL,
    updated_at   TIMESTAMPTZ   DEFAULT NOW()
);

-- ============================================================================
-- 10. UNIVERSES — Named collections of symbols
-- ============================================================================

CREATE TABLE backtester.universes (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name         TEXT UNIQUE NOT NULL,
    description  TEXT,
    type         TEXT NOT NULL DEFAULT 'static',  -- 'static','filter','index'
    definition   JSONB NOT NULL DEFAULT '{}',     -- filter rules or index name
    symbol_count INTEGER DEFAULT 0,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- 11. UNIVERSE_SYMBOLS — Many-to-many: universe <-> symbol membership
-- ============================================================================

CREATE TABLE backtester.universe_symbols (
    universe_id UUID NOT NULL REFERENCES backtester.universes(id) ON DELETE CASCADE,
    symbol      TEXT NOT NULL,
    added_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (universe_id, symbol)
);

CREATE INDEX ix_universe_symbols_sym ON backtester.universe_symbols (symbol);

-- ============================================================================
-- 12. INDICATORS_CACHE — Pre-computed indicator values
-- ============================================================================
-- Caches expensive indicator computations keyed by indicator+config+data hash.

CREATE TABLE backtester.indicators_cache (
    symbol         TEXT          NOT NULL,
    timeframe      TEXT          NOT NULL,
    indicator_name TEXT          NOT NULL,
    config_hash    TEXT          NOT NULL,
    ts             TIMESTAMPTZ   NOT NULL,
    value          NUMERIC(15,6),
    values_json    JSONB,        -- for multi-output indicators (e.g. BB upper/mid/lower)
    data_hash      TEXT          NOT NULL,
    created_at     TIMESTAMPTZ   DEFAULT NOW(),

    PRIMARY KEY (symbol, timeframe, indicator_name, config_hash, ts)
);

SELECT create_hypertable('backtester.indicators_cache', 'ts');

CREATE INDEX ix_indcache_lookup
    ON backtester.indicators_cache (symbol, timeframe, indicator_name, config_hash, ts DESC);

-- ============================================================================
-- 13. OPTIMIZATION_RUNS — Parameter optimization jobs
-- ============================================================================
-- Tracks a parameter sweep / Bayesian / genetic optimization session.

CREATE TABLE backtester.optimization_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_id     UUID REFERENCES backtester.backtests(id),  -- parent backtest
    strategy_name   TEXT          NOT NULL,
    method          TEXT          NOT NULL DEFAULT 'bayesian',
        -- 'grid','random','bayesian','genetic'
    objective       TEXT          NOT NULL DEFAULT 'sharpe_ratio',
    param_space     JSONB         NOT NULL,   -- search space definition
    total_trials    INTEGER       NOT NULL,
    completed_trials INTEGER      DEFAULT 0,
    best_params     JSONB,
    best_objective  NUMERIC(10,4),
    status          TEXT          NOT NULL DEFAULT 'pending',
    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

-- ============================================================================
-- 14. OPTIMIZATION_RESULTS — Individual trial results
-- ============================================================================

CREATE TABLE backtester.optimization_results (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    optimization_id  UUID NOT NULL REFERENCES backtester.optimization_runs(id) ON DELETE CASCADE,
    trial_number     INTEGER       NOT NULL,
    params           JSONB         NOT NULL,
    param_hash       TEXT          NOT NULL,
    objective_value  NUMERIC(10,4),
    metrics          JSONB,         -- full metrics snapshot
    overfitting_score NUMERIC(4,3),
    duration_seconds NUMERIC(10,3),
    created_at       TIMESTAMPTZ   DEFAULT NOW()
);

CREATE INDEX ix_optresults_run ON backtester.optimization_results (optimization_id, trial_number);

-- ============================================================================
-- Supporting tables
-- ============================================================================

-- Symbol metadata
CREATE TABLE backtester.symbols (
    symbol              TEXT PRIMARY KEY,
    company_name        TEXT,
    sector              TEXT,
    industry            TEXT,
    market_cap          BIGINT,
    exchange            TEXT,
    currency            TEXT DEFAULT 'USD',
    is_active           BOOLEAN DEFAULT TRUE,
    first_price_date    DATE,
    last_price_date     DATE,
    available_timeframes TEXT[] DEFAULT '{1d}',
    avg_daily_volume    BIGINT,
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Data freshness tracking
CREATE TABLE backtester.data_freshness (
    symbol         TEXT NOT NULL,
    timeframe      TEXT NOT NULL,
    last_update    TIMESTAMPTZ NOT NULL,
    last_timestamp TIMESTAMPTZ NOT NULL,
    data_source    TEXT NOT NULL,
    record_count   BIGINT NOT NULL,
    PRIMARY KEY (symbol, timeframe)
);

-- Corporate actions
CREATE TABLE backtester.corporate_actions (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol       TEXT NOT NULL,
    ex_date      DATE NOT NULL,
    action_type  TEXT NOT NULL,  -- 'split','dividend','spinoff'
    ratio        NUMERIC(10,4),
    amount       NUMERIC(8,4),
    description  TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (symbol, ex_date, action_type)
);

-- ============================================================================
-- Done. All tables created.
-- ============================================================================
