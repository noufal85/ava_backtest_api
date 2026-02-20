# V2 Architecture Design

## Executive Summary

**Trading Backtester V2** is a ground-up rewrite designed for speed, composability, and correctness. It maintains Python as the primary language while addressing V1's architectural limitations through a modular, event-driven design that prevents look-ahead bias by construction.

## Core Design Principles

1. **Composable Architecture** — Strategies, position sizing, risk management, and execution are separate pluggable components
2. **No Look-Ahead by Construction** — Data flows enforce temporal constraints at the API level
3. **Fast Vectorized Operations** — NumPy/Pandas operations with optional Polars for large datasets
4. **Multi-Timeframe Native** — Built-in support for mixed timeframes without complexity
5. **Observable & Reproducible** — Every decision is logged and traceable
6. **Test-First Design** — Easy to unit test strategies in isolation

## Module Layout

```
v2/
├── core/
│   ├── data/                 # Data layer abstraction
│   │   ├── providers/        # Data source adapters (FMP, Alpha Vantage, etc.)
│   │   ├── cache/           # Smart caching layer
│   │   ├── universe/        # Universe management
│   │   └── validator/       # Data quality checks
│   ├── strategy/            # Strategy framework
│   │   ├── base.py          # New base strategy interface
│   │   ├── signals.py       # Signal generation helpers
│   │   ├── versioning.py    # Strategy versioning system
│   │   └── converter/       # Pine Script → Python converter
│   ├── execution/           # Trade execution simulation
│   │   ├── engine.py        # Main backtest engine
│   │   ├── fills.py         # Fill simulation (slippage, partial fills)
│   │   ├── costs.py         # Commission and fee models
│   │   └── portfolio.py     # Portfolio state management
│   ├── sizing/              # Position sizing components
│   │   ├── fixed.py         # Fixed dollar sizing
│   │   ├── kelly.py         # Kelly criterion
│   │   ├── risk_parity.py   # Risk-based sizing
│   │   └── volatility.py    # Vol-adjusted sizing
│   ├── risk/                # Risk management middleware
│   │   ├── stops.py         # Stop-loss implementations
│   │   ├── limits.py        # Position and exposure limits
│   │   └── drawdown.py      # Drawdown controls
│   ├── analytics/           # Performance analysis
│   │   ├── metrics.py       # Performance metrics
│   │   ├── attribution.py   # Trade-level attribution
│   │   └── optimization/    # Parameter optimization
│   └── indicators/          # Technical indicators
│       ├── trend.py         # Moving averages, MACD, etc.
│       ├── momentum.py      # RSI, Stochastic, etc.
│       ├── volatility.py    # Bollinger Bands, ATR, etc.
│       └── volume.py        # Volume-based indicators
├── strategies/              # Strategy implementations
│   ├── classic/             # Traditional strategies (MA cross, etc.)
│   ├── modern/              # Advanced strategies
│   └── experimental/        # Research strategies
├── api/                     # FastAPI application
│   ├── v2/                  # V2 endpoints
│   ├── ws/                  # WebSocket handlers
│   └── middleware/          # CORS, auth, etc.
├── ui/                      # React frontend (separate repo)
├── cli/                     # Command-line interface
├── tests/                   # Comprehensive test suite
└── migrations/              # Database schema evolution
```

## Data Flow

### High-Level Pipeline

```
1. Data Ingestion → TimescaleDB (OHLCV + metadata)
2. Strategy Execution:
   a. Load Price Data (with lookback buffer)
   b. Compute Indicators (vectorized)
   c. Generate Signals (bar-by-bar, no lookahead)
   d. Size Positions (pluggable sizers)
   e. Apply Risk Rules (middleware)
   f. Execute Trades (realistic fills)
   g. Track Portfolio State
3. Analytics → Metrics, Attribution, Visualization
4. Storage → Results, trades, equity curves to TimescaleDB
```

### Temporal Data Flow (Lookahead Prevention)

```
Historical Data Stream:
[Bar 1] → [Bar 2] → [Bar 3] → ... → [Bar N]
    ↓         ↓         ↓
Indicator    Signal    Risk     Portfolio
Pipeline  → Pipeline → Rules  → Update
    ↓         ↓         ↓         ↓
[I₁]     → [S₁]     → [R₁]   → [P₁]
```

**Key Innovation**: Each stage only has access to data ≤ current bar. The `DataWindow` class enforces this at runtime.

## Technology Choices

### Core Stack (Keep from V1)
- **Python 3.11+** — Noufal's preference, mature ecosystem
- **FastAPI** — Async REST API with WebSocket support
- **TimescaleDB** — Time-series optimized PostgreSQL for OHLCV data
- **Docker** — Containerized deployment

### Data Processing (Upgrade)
- **Polars** (primary) — 10-100x faster than Pandas for large datasets
- **Pandas** (compatibility) — For existing indicator library, gradual migration
- **NumPy** — Vectorized computation backbone
- **AsyncIO** — Parallel symbol processing

### New Additions
- **Pydantic V2** — Configuration and data validation
- **Ruff** — Lightning-fast linting and formatting
- **pytest-xdist** — Parallel test execution
- **Redis** — Caching layer for indicators and interim results
- **Prometheus + Grafana** — Monitoring and observability

## Database Schema Evolution

### Current Schema Issues (V1)
- No strategy versioning
- Limited parameter tracking
- No trade attribution metadata
- No support for walk-forward results
- No correlation analysis storage

### V2 Schema Improvements

```sql
-- Enhanced strategy runs with versioning
CREATE TABLE backtester.strategy_runs_v2 (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name TEXT NOT NULL,
    strategy_version TEXT NOT NULL,        -- semantic version
    strategy_hash TEXT NOT NULL,           -- code + config hash
    param_hash TEXT NOT NULL,
    param_yaml TEXT NOT NULL,
    run_type TEXT NOT NULL DEFAULT 'backtest', -- backtest, walk_forward, monte_carlo
    parent_run_id UUID REFERENCES backtester.strategy_runs_v2(id), -- for optimization runs
    universe_name TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(15,2) NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_seconds DECIMAL(10,3),
    symbols_processed INTEGER,
    trades_count INTEGER,
    error_message TEXT,
    metadata JSONB
);

-- Enhanced trades with attribution
CREATE TABLE backtester.trades_v2 (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES backtester.strategy_runs_v2(id),
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    entry_date DATE NOT NULL,
    entry_price DECIMAL(12,4) NOT NULL,
    entry_time TIME, -- for intraday strategies
    exit_date DATE,
    exit_price DECIMAL(12,4),
    exit_time TIME,
    shares INTEGER NOT NULL,
    pnl DECIMAL(15,2),
    pnl_pct DECIMAL(8,4),
    commission_cost DECIMAL(10,2),
    slippage_cost DECIMAL(10,2),
    borrowing_cost DECIMAL(10,2), -- for shorts
    entry_signal JSONB,
    exit_reason TEXT,
    hold_days INTEGER,
    max_favorable_excursion DECIMAL(15,2),
    max_adverse_excursion DECIMAL(15,2),
    -- Attribution fields
    regime_at_entry TEXT,
    regime_confidence DECIMAL(4,3),
    volatility_at_entry DECIMAL(8,4),
    volume_at_entry BIGINT,
    -- Risk metrics
    position_size_pct DECIMAL(6,3),
    portfolio_heat DECIMAL(6,3), -- total exposure when trade opened
    drawdown_at_entry DECIMAL(6,3),
    -- Execution quality
    fill_quality_score DECIMAL(4,3), -- how realistic the fill was
    market_impact DECIMAL(8,4)
);

-- Hypertable partitioning by entry_date
SELECT create_hypertable('backtester.trades_v2', 'entry_date');

-- Walk-forward analysis results
CREATE TABLE backtester.walkforward_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_run_id UUID NOT NULL REFERENCES backtester.strategy_runs_v2(id),
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    train_start DATE NOT NULL,
    train_end DATE NOT NULL,
    test_start DATE NOT NULL,
    test_end DATE NOT NULL,
    optimal_params JSONB NOT NULL,
    is_metrics DECIMAL(8,4), -- in-sample Sharpe
    oos_metrics DECIMAL(8,4), -- out-of-sample Sharpe
    degradation_pct DECIMAL(6,2), -- (IS - OOS) / IS * 100
    trades_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Parameter optimization tracking
CREATE TABLE backtester.optimization_trials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES backtester.strategy_runs_v2(id),
    trial_number INTEGER NOT NULL,
    params JSONB NOT NULL,
    param_hash TEXT NOT NULL,
    objective_value DECIMAL(10,4), -- what we're optimizing for
    metrics JSONB, -- all computed metrics
    overfitting_score DECIMAL(4,3), -- measure of parameter overfitting
    completed_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Dependency Graph

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Indicator Lib  │    │  Risk Manager   │
│  (providers,    │    │  (vectorized    │    │  (stops, limits,│
│   cache, etc.)  │    │   functions)    │    │   drawdown)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Strategy Framework                           │
│  (base classes, signal generation, temporal enforcement)        │
└─────────────────────────────────────────────────────────────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────┐                           ┌─────────────────┐
│ Position Sizers │                           │ Execution Engine│
│ (kelly, fixed,  │                           │ (fills, costs,  │
│  risk-based)    │                           │  portfolio)     │
└─────────────────┘                           └─────────────────┘
         │                                               │
         └───────────────────┐       ┌───────────────────┘
                             ▼       ▼
                    ┌─────────────────┐
                    │   Analytics     │
                    │ (metrics, attr, │
                    │  optimization)  │
                    └─────────────────┘
```

## What's Reusable from V1

### Keep & Enhance
- **Trade Model** — Excellent foundation, add attribution fields
- **Config System** — Pydantic validation is solid, add strategy versioning
- **Core Indicators** — Math is correct, but vectorize more efficiently
- **Database Models** — Good schema, extend with V2 tables
- **Cost Models** — Slippage and commission logic works well
- **Regime Integration** — Novel feature, enhance with more sophisticated classifiers

### Enhance Significantly
- **Strategy Pattern** — Too rigid, make more composable
- **Engine** — Too monolithic, separate concerns
- **API Layer** — Add WebSocket progress, better error handling
- **Data Loading** — Cache aggressively, lazy loading
- **Testing** — More comprehensive strategy testing framework

### Rewrite from Scratch
- **Multi-Timeframe Logic** — V1 is hacky, V2 should be native
- **Position Sizing** — Currently embedded in strategies, should be pluggable
- **Risk Management** — Currently config-driven, should be middleware
- **Optimization Framework** — V1 has none, V2 needs robust parameter optimization
- **UI Integration** — V1 has basic endpoints, V2 needs real-time updates

## Key Innovations in V2

### 1. Temporal Data Windows
```python
# V1: Full DataFrame access (lookahead possible)
def generate_signals(self, df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    return df.copy()  # Can accidentally access future data

# V2: Enforced temporal windows
class DataWindow:
    def __init__(self, df: pl.DataFrame, current_bar: int):
        self._df = df
        self._current = current_bar
    
    def indicators(self) -> pl.DataFrame:
        """Access to current + historical bars for indicator computation."""
        return self._df[:self._current + 1]
    
    def current(self) -> pl.DataFrame:
        """Access to only current bar for signal generation."""
        return self._df[self._current:self._current + 1]
    
    def historical(self, lookback: int = None) -> pl.DataFrame:
        """Access to historical bars only (no current bar)."""
        end_idx = self._current
        start_idx = max(0, end_idx - (lookback or end_idx))
        return self._df[start_idx:end_idx]
```

### 2. Middleware Pipeline
```python
# Strategy execution becomes a pipeline
pipeline = [
    IndicatorMiddleware(strategy.indicators),
    SignalMiddleware(strategy.signals),
    SizingMiddleware(kelly_sizer),
    RiskMiddleware(stop_loss_manager),
    ExecutionMiddleware(realistic_fills),
    AttributionMiddleware(trade_logger)
]
```

### 3. Plugin Architecture
```python
# Everything is pluggable
strategy = StrategyBuilder() \
    .with_signals(BollingerMeanReversion()) \
    .with_sizing(KellyOptimal(risk_free_rate=0.05)) \
    .with_risk(TrailingStop(pct=0.02)) \
    .with_regime(RegimeFilter(['bull', 'neutral'])) \
    .build()
```

## Performance Targets

### V1 Baseline (measured)
- Single symbol backtest: ~2-3 seconds
- 100 symbols (4 years): ~5-10 minutes
- Memory usage: ~500MB for large backtests

### V2 Goals
- **10x faster** single symbol: ~200ms
- **5x faster** multi-symbol: ~1-2 minutes for 100 symbols
- **50% less memory** through lazy loading and Polars efficiency
- **Parallel optimization**: 1000 parameter combinations in <30 minutes
- **Real-time progress**: WebSocket updates every 5% completion

## Database Design (TimescaleDB)

### Optimized Hypertables
```sql
-- Price data with compression
CREATE TABLE prices_v2 (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timeframe TEXT NOT NULL, -- '1d', '15m', '5m', '1m'
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume BIGINT,
    adj_close DECIMAL(12,4),
    split_factor DECIMAL(8,4) DEFAULT 1.0,
    dividend DECIMAL(8,4) DEFAULT 0.0
);

SELECT create_hypertable('prices_v2', 'timestamp');
SELECT add_dimension('prices_v2', 'symbol', number_partitions => 4);

-- Enable compression for older data
ALTER TABLE prices_v2 SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe',
    timescaledb.compress_orderby = 'timestamp DESC'
);
```

### Indicator Caching
```sql
-- Computed indicators cache (massive speedup)
CREATE TABLE indicator_cache (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    indicator_name TEXT NOT NULL,
    config_hash TEXT NOT NULL, -- indicator parameters hash
    data_hash TEXT NOT NULL,   -- underlying price data hash
    timestamp TIMESTAMPTZ NOT NULL,
    value DECIMAL(15,6),
    metadata JSONB
);

SELECT create_hypertable('indicator_cache', 'timestamp');
```

## Configuration Management

### Strategy Versioning
```yaml
# V2 config format
strategy: bb_mean_reversion
version: "2.1.0"
runtime:
  candle_interval: "15m"
  warmup_periods: 100    # bars needed before first signal
  
components:
  indicators:
    - bollinger_bands: {period: 20, std_dev: 2.0}
    - rsi: {period: 14}
    
  signals:
    entry_long:
      - bb_lower_touch: {}
      - rsi_oversold: {threshold: 30}
    exit_long:
      - bb_upper_touch: {}
      
  sizing:
    type: kelly_optimal
    params: {risk_free_rate: 0.05, max_leverage: 1.0}
    
  risk:
    - trailing_stop: {pct: 0.02}
    - max_position: {pct: 0.05}
    - max_correlation: {threshold: 0.7}
    
  regime:
    enabled: true
    allowed: ["bull", "neutral"]
    confidence_threshold: 0.8

backtest:
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  initial_capital: 100000.0
  universe: "sp500_liquid"
  
execution:
  fill_model: "realistic_volume"
  commission_model: "interactive_brokers"
  slippage_model: "sqrt_volume"
```

## Multi-Timeframe Architecture

### V1 Problem
- Indicator timeframe separate from candle timeframe
- Manual data alignment
- Easy to introduce lookahead bugs

### V2 Solution: Native Multi-Timeframe
```python
class MultiTimeframeData:
    def __init__(self):
        self.timeframes = {}  # "15m": DataFrame, "1d": DataFrame
        self.current_time = None
        
    def align_to_primary(self, primary_tf: str) -> None:
        """Align all timeframes to primary timeframe bars."""
        primary_bars = self.timeframes[primary_tf]
        for tf, df in self.timeframes.items():
            if tf != primary_tf:
                # Forward-fill alignment (no lookahead)
                aligned = self._align_timeframe(df, primary_bars, tf, primary_tf)
                self.timeframes[f"{tf}_aligned"] = aligned
                
    def current_bar(self, timeframe: str) -> pl.DataFrame:
        """Get current bar for any timeframe (properly aligned)."""
        return self.timeframes[f"{timeframe}_aligned"].slice(self.current_idx, 1)
```

## Error Handling & Observability

### Structured Logging
```python
# Every component uses structured logging
logger.info(
    "trade_executed",
    strategy=strategy.name,
    symbol=trade.symbol,
    direction=trade.direction,
    entry_price=trade.entry_price,
    position_size=trade.shares,
    portfolio_heat=portfolio.heat_pct(),
    regime=current_regime,
    signal_strength=signal.strength
)
```

### Health Monitoring
- Strategy execution metrics (trades/sec, memory usage)
- Data pipeline health (missing data, staleness)
- API performance (response times, error rates)
- Background job monitoring (data sync status)

## Migration Strategy from V1

### Phase 1: Parallel Development
- V2 developed alongside V1 (no disruption)
- Shared TimescaleDB with separate schemas
- V1 continues handling production backtests

### Phase 2: Strategy Migration
- Automated conversion tool for V1 → V2 strategy format
- Validation: run same strategy on both versions, compare results
- Gradual migration of high-value strategies

### Phase 3: Cutover
- V2 becomes primary system
- V1 endpoints marked deprecated
- Data export tool for V1 results

## Success Metrics

### Performance
- [ ] 10x faster single-symbol backtests
- [ ] 5x faster multi-symbol backtests
- [ ] <2GB memory usage for largest historical backtests
- [ ] <5% CPU usage during idle

### Reliability
- [ ] Zero look-ahead bias (verified by extensive testing)
- [ ] 99.9% uptime for API
- [ ] <1% data quality issues
- [ ] Deterministic results (same config = same output)

### Developer Experience
- [ ] <10 lines to implement a basic strategy
- [ ] <5 minutes to add new indicator
- [ ] 100% test coverage for critical path
- [ ] <30 seconds for full test suite

### Business Value
- [ ] Support for 1000+ simultaneous backtests
- [ ] Real-time optimization (parameter sweeps in <10 minutes)
- [ ] Walk-forward analysis for robust strategy validation
- [ ] Easy Pine Script conversion (target: 90% success rate)

This architecture positions V2 as a production-grade quantitative research platform rather than just a backtesting tool.