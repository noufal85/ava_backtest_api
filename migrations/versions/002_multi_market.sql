ALTER TABLE backtester.candles_daily ADD COLUMN IF NOT EXISTS market_code TEXT NOT NULL DEFAULT 'US';
ALTER TABLE backtester.candles_intraday ADD COLUMN IF NOT EXISTS market_code TEXT NOT NULL DEFAULT 'US';
ALTER TABLE backtester.backtests ADD COLUMN IF NOT EXISTS market_code TEXT NOT NULL DEFAULT 'US';
ALTER TABLE backtester.backtests ADD COLUMN IF NOT EXISTS currency TEXT NOT NULL DEFAULT 'USD';
ALTER TABLE backtester.backtest_results ADD COLUMN IF NOT EXISTS stt_total NUMERIC(15,2) DEFAULT 0;
ALTER TABLE backtester.backtest_results ADD COLUMN IF NOT EXISTS gst_total NUMERIC(15,2) DEFAULT 0;
ALTER TABLE backtester.backtest_results ADD COLUMN IF NOT EXISTS stamp_duty NUMERIC(15,2) DEFAULT 0;
ALTER TABLE backtester.trades ADD COLUMN IF NOT EXISTS stt_cost NUMERIC(10,2) DEFAULT 0;
ALTER TABLE backtester.trades ADD COLUMN IF NOT EXISTS gst_cost NUMERIC(10,2) DEFAULT 0;
ALTER TABLE backtester.trades ADD COLUMN IF NOT EXISTS stamp_duty NUMERIC(10,2) DEFAULT 0;
ALTER TABLE backtester.symbols ADD COLUMN IF NOT EXISTS market_code TEXT NOT NULL DEFAULT 'US';
ALTER TABLE backtester.universes ADD COLUMN IF NOT EXISTS market_code TEXT NOT NULL DEFAULT 'US';
CREATE INDEX IF NOT EXISTS ix_backtests_market ON backtester.backtests (market_code, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_universes_market ON backtester.universes (market_code);
INSERT INTO backtester.universes (name, description, type, market_code) VALUES
  ('nifty50', 'Nifty 50 index', 'index', 'IN'),
  ('nifty100', 'Nifty 100 index', 'index', 'IN'),
  ('nifty500', 'Nifty 500 index', 'index', 'IN'),
  ('nse_large_cap', 'NSE Large Cap', 'filter', 'IN'),
  ('nse_mid_cap', 'NSE Mid Cap', 'filter', 'IN'),
  ('nse_small_cap', 'NSE Small Cap', 'filter', 'IN')
ON CONFLICT DO NOTHING;
