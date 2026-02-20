# Trading Backtester V1: Comprehensive Analysis of Shortcomings & Lessons Learned

*Analysis Date: 2026-02-20*  
*Analyst: Ava (V2 Design Team)*

## Executive Summary

The trading backtester v1 represents a significant first iteration with ~69 strategies, FastAPI backend, React UI, and Docker deployment. However, several critical architectural and operational issues have emerged that necessitate a v2 redesign. This analysis documents every identified shortcoming to guide the v2 architecture.

**Key Findings:**
- **2 of 5 TV strategies fail** with "no price data found" despite data existing
- **Long+short mode destroys returns** for ALL strategies 
- **Look-ahead bias present** in multiple strategies
- **Monolithic architecture** makes debugging and extension difficult
- **Poor error handling** leads to silent failures
- **Configuration drift** (kell_base_breakout: PF 2.92 → 0.81)
- **Intraday framework fundamental issues** - all intraday backtests lost money

---

## 1. Architecture Issues

### 1.1 Monolithic Design Problems

**Issue:** Single large engine orchestrates everything from data loading to P&L computation.
```python
# BacktestEngine.run() does EVERYTHING:
# - Symbol resolution, price loading, indicator computation
# - Signal generation, trade execution, cost application  
# - MFE/MAE calculation, equity curve, metrics, DB persistence
```

**Problems:**
- **No separation of concerns** - impossible to test individual stages
- **Failure cascades** - one symbol error kills entire backtest
- **Poor debuggability** - can't isolate where things break
- **Hard to extend** - adding new features requires modifying the monolith
- **Resource waste** - processes all symbols sequentially per stage instead of true parallelism

**Impact:** When TV strategies fail with "no price data found", impossible to determine if it's:
- Data loading issue  
- Symbol resolution problem
- Date range filtering bug
- Database query problem

### 1.2 Tight Coupling

**Issue:** Core components are tightly coupled, making testing and mocking difficult.

**Examples:**
- `BaseStrategy` directly calls database-dependent methods
- `BacktestEngine` mixes business logic with DB persistence
- Strategies depend on specific DataFrame column names and structures
- No dependency injection - everything is hardcoded

**Impact:** 
- Unit testing requires full database setup
- Can't test strategies against mock data easily
- Changing data format breaks all strategies
- No way to run strategies independently of the full framework

### 1.3 Poor Error Handling

**Issue:** Errors propagate poorly and often get silently swallowed.

```python
except Exception:
    logger.exception("Failed processing symbol %s — skipping", sym)
    return None  # Silent skip - no visibility into failure
```

**Problems:**
- **Silent failures** - strategies that generate 0 trades (tv_atr_trailing_stop)
- **Poor error context** - "no price data found" doesn't say which component failed
- **Exception swallowing** - symbol processing failures are logged but not exposed to API
- **No error aggregation** - can't see failure patterns across runs

### 1.4 No Clear Boundaries

**Issue:** Responsibilities are mixed across layers.

**Examples:**
- `BaseStrategy.filter_universe()` does data filtering (should be data layer)
- `BacktestEngine` computes MFE/MAE (should be trade analysis layer)  
- API routes contain business logic (portfolio simulation)
- Database models mixed with business logic

## 2. Data Layer Problems

### 2.1 Inconsistent Data Loading

**Issue:** Multiple data loading patterns with different failure modes.

```python
# Daily vs Intraday loading have different schemas/error handling
async def _load_daily_prices(...) -> pd.DataFrame:
async def _load_intraday_prices(...) -> pd.DataFrame:
```

**Problems:**
- **Different column names** between daily and intraday data
- **No unified data interface** - strategies must handle multiple formats
- **Inconsistent date handling** - mix of date vs datetime objects
- **No data validation** - GIGO (garbage in, garbage out)

### 2.2 Poor Data Validation

**Issue:** No validation of price data integrity before running backtests.

**Missing checks:**
- **Price sanity checks** (negative prices, extreme outliers)
- **Volume validation** (zero volume days, impossible volumes)
- **Date gap detection** (missing trading days)
- **Adjusted close alignment** (splits/dividends handled correctly?)
- **Data completeness** (minimum bars required per symbol)

**Impact:** Strategies fail with cryptic errors or produce invalid results.

### 2.3 Caching Issues

**Issue:** No data caching leads to repeated DB queries for the same data.

**Problems:**
- **Symbol universe** loaded repeatedly across runs
- **Price data** re-queried for overlapping backtests
- **Indicator computation** repeated even when params unchanged
- **Memory inefficient** - each symbol processed in isolation

### 2.4 Multi-Timeframe Confusion

**Issue:** `candle_interval` vs `indicator_interval` creates confusion and bugs.

```python
# These can be different - but how do you align them properly?
candle_interval: str = "15min"
indicator_interval: str = "1d"  # Daily indicators on intraday data
```

**Problems:**
- **Alignment issues** - how to map daily indicators to intraday bars?
- **Look-ahead potential** - using future daily data for intraday decisions
- **Performance overhead** - loading two timeframes per symbol
- **Strategy complexity** - strategies must handle both timeframes

## 3. Strategy Pattern Limitations

### 3.1 Three-Method Constraint

**Issue:** All strategies forced into 3-method pattern:
1. `compute_indicators()`
2. `generate_signals()`  
3. `execute_trades()`

**Limitations:**
- **No lifecycle hooks** - can't initialize per-symbol state
- **No position management** - can't track open positions across bars
- **No dynamic exits** - exits must be computed upfront, not adjusted based on current state
- **No inter-bar state** - each bar processed independently
- **Forced batch processing** - can't stream/react to new data

### 3.2 Static Signal Generation

**Issue:** Signals must be computed for entire DataFrame upfront.

```python
def generate_signals(self, df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    # Must return signals for ALL bars at once
    # Can't adjust based on current portfolio state
    # Can't react to fills or partial fills
```

**Problems:**
- **No dynamic position sizing** based on current portfolio
- **No risk management** that considers current exposure
- **No adaptive stops** that adjust based on market conditions
- **No portfolio-aware signals** - each symbol processed independently

### 3.3 Limited Trade Types

**Issue:** Trade model only supports simple entry/exit pairs.

**Missing:**
- **Scale-in/scale-out** - multiple entries/exits per "trade"
- **Option strategies** - spreads, straddles, covered calls
- **Partial fills** - realistic order execution
- **Order types** - stop orders, limit orders, bracket orders
- **Position adjustments** - adding to winners, trimming losers

### 3.4 Poor Strategy Debugging

**Issue:** When strategies fail or generate unexpected results, very hard to debug.

**Missing:**
- **Indicator visualization** - can't plot intermediate calculations
- **Signal debugging** - can't see why signals fired or didn't fire
- **Trade attribution** - can't trace trade decision back to signal logic
- **Step-by-step execution** - can't pause/inspect state between bars

## 4. Execution Engine Flaws

### 4.1 Unrealistic Fill Simulation

**Issue:** All trades filled at exact close price with simple slippage model.

**Problems:**
- **No market impact** modeling for large orders
- **No liquidity constraints** - can "buy" $100K of penny stock instantly
- **No gap handling** - fills through overnight gaps at previous close
- **No partial fills** - orders either fill 100% or not at all
- **Fixed slippage** - doesn't vary by volatility, time of day, or order size

### 4.2 Portfolio Simulation Issues

**Issue:** `PortfolioTracker` oversimplified and has bugs.

```python
def can_open_position(...) -> bool:
    # Checks buying power but not margin requirements
    # No position correlation risk
    # No portfolio heat calculation
```

**Problems:**
- **No margin modeling** - treats everything as cash account
- **Position sizing bugs** - max_position_pct can be violated
- **No correlation risk** - can hold 10 highly correlated positions
- **Cash calculation errors** - doesn't handle costs properly in some cases
- **No intraday position tracking** - for intraday strategies

### 4.3 Cost Model Oversimplification

**Issue:** Commission and slippage models too basic.

**Missing:**
- **Market maker rebates** for providing liquidity
- **Exchange fees** vary by venue (NASDAQ vs NYSE)
- **Account size tiers** - commissions decrease with volume
- **Time-of-day effects** - spreads wider at open/close
- **Crypto exchange fees** - different fee structures entirely

### 4.4 Long+Short Mode Fundamental Flaw

**Issue:** Long+short mode destroys returns for ALL strategies.

**Root causes (suspected):**
- **Symmetry assumption** - assumes short side mirrors long side exactly
- **Borrowing costs ignored** - no short interest/borrow fees
- **Timing issues** - short signals may be lagged or inverted poorly
- **Risk model mismatch** - stop losses may work differently for shorts
- **Market bias** - markets trend up over time, shorts fight the trend

**Evidence:** Only 6/24 strategies survived long+short mode, all others went negative.

## 5. API/UI Issues

### 5.1 Async Background Tasks Don't Work

**Issue:** Strategy optimizer couldn't run backtests due to API issues.

```python
async def _run_backtest(run_id: uuid.UUID, strategy_name: str, config_path: str, overrides: dict | None):
    # Background task has inconsistent error handling
    # Database sessions may conflict between main thread and background
```

**Problems:**
- **Session conflicts** - async DB sessions not properly isolated
- **No task progress tracking** - can't tell if background job is actually running
- **Poor error propagation** - errors in background tasks lost
- **No cancellation** - can't stop runaway backtests
- **Resource leaks** - orphaned tasks may continue running

### 5.2 Slow Query Performance

**Issue:** Complex queries for runs/trades/signals are slow.

**Problems:**
- **Missing indexes** on commonly queried columns
- **N+1 query problems** - separate query for metrics for each run
- **Large result sets** - no streaming for big trade lists
- **No query optimization** - joins not optimized
- **No caching** - same queries repeated constantly

### 5.3 Poor Error Messages

**Issue:** API returns cryptic errors that don't help users.

**Examples:**
- "No price data found" - doesn't specify which symbol/date range
- "Invalid configuration" - doesn't specify which field is invalid
- Generic 500 errors instead of meaningful HTTP codes
- No suggestion for fixing common problems

### 5.4 Missing Real-Time Features

**Issue:** Everything is batch-oriented, no real-time capabilities.

**Missing:**
- **Live strategy monitoring** - can't see current positions/signals
- **Real-time alerts** - no notifications when signals fire
- **Live P&L tracking** - no real-time equity curve updates
- **Hot strategy swapping** - can't update strategies without restart

## 6. Testing Gaps

### 6.1 Low Test Coverage

**Current state:** 13 test files covering core functionality only.

**Missing tests:**
- **Strategy-specific tests** - each strategy should have unit tests
- **Integration tests** - end-to-end backtest workflows  
- **Data quality tests** - verify price data integrity
- **Performance tests** - ensure backtests complete within time bounds
- **Error handling tests** - verify graceful failure modes

### 6.2 Mock Data Problems

**Issue:** Tests use simplified mock data that doesn't represent real market conditions.

**Problems:**
- **No gap scenarios** - overnight gaps, earnings gaps
- **No missing data** - real markets have holidays, halted stocks
- **No extreme scenarios** - crashes, volatility spikes, illiquid symbols
- **Perfect data** - tests pass on clean data but fail on real data

### 6.3 No Regression Testing

**Issue:** When kell_base_breakout degraded from PF 2.92 → 0.81, no tests caught it.

**Missing:**
- **Performance regression tests** - alert when strategy performance changes significantly
- **Config validation tests** - prevent parameter drift
- **Data drift detection** - detect when underlying data changes
- **Historical result preservation** - golden masters for strategy output

## 7. Performance Bottlenecks

### 7.1 Single-Threaded Strategy Processing

**Issue:** Despite async framework, strategy execution is essentially single-threaded per symbol.

```python
# Each symbol processed sequentially through all stages
df = strategy.compute_indicators(df, config, daily_data=daily_df)
signals_df = strategy.generate_signals(df, config)
trades = strategy.execute_trades(df, signals_df, config)
```

**Problems:**
- **CPU underutilization** - only uses one core per symbol
- **Memory inefficient** - loads full DataFrames for each symbol
- **No pipeline parallelism** - can't overlap indicator computation with signal generation
- **Blocking I/O** - database queries block other processing

### 7.2 Memory Issues

**Issue:** Large backtests (multi-year, 100+ symbols) hit memory limits.

**Problems:**
- **Full DataFrame loading** - entire price history loaded per symbol
- **No lazy evaluation** - all indicators computed upfront
- **Equity curve explosion** - daily equity tracking for all symbol combinations
- **Signal storage** - stores every signal even if "hold"

### 7.3 Database Performance

**Issue:** Bulk inserts and queries are inefficient.

```python
# Bulk insert with 1000-row batches - arbitrary limit
BATCH = 1000
for i in range(0, len(all_signals), BATCH):
```

**Problems:**
- **Fixed batch sizes** don't optimize for different data sizes
- **No connection pooling** - new connection per operation
- **Missing indexes** on query-heavy columns
- **No query plan analysis** - slow queries not identified

## 8. Configuration Issues

### 8.1 YAML Configuration Sprawl

**Issue:** 69 strategies = 69+ YAML files with complex nested structure.

**Problems:**
- **Parameter drift** - configs change without tracking (kell_base_breakout example)
- **Validation gaps** - invalid configs only caught at runtime
- **No version control** - can't rollback config changes
- **Duplication** - similar strategies have near-identical configs
- **No templates** - hard to create new strategy configs

### 8.2 Poor Parameter Management

**Issue:** Strategy parameters hardcoded in multiple places.

```python
# In strategy code:
atr_length = int(params.get("atr_length", 14))  # Default here
# In YAML config:
atr_length: 14  # Default also here
# In UI:
// Default also hardcoded in React forms
```

**Problems:**
- **Defaults scattered** across code, config, and UI
- **No parameter validation** - can set nonsensical values
- **No parameter ranges** - no min/max constraints
- **No dependencies** - can't express "if X then Y must be Z"

### 8.3 Configuration Hash Issues

**Issue:** `compute_param_hash()` is fragile and doesn't capture all relevant state.

**Missing from hash:**
- **Code version** - same config, different strategy implementation
- **Data version** - same config, different underlying data  
- **Environment** - development vs production differences

**Impact:** Config changes may not trigger re-runs when they should.

## 9. Deployment/Docker Problems

### 9.1 Stale Code Problem

**Issue:** Docker builds cache outdated Python code.

```dockerfile
# pip install -e . in Docker = stale code
RUN pip install --no-cache-dir -e .
```

**From live-trader lesson:** `pip install -e .` in Docker can serve stale code because the source directory is bind-mounted. Changed to `pip install .` without -e flag.

**Impact:** Code changes don't take effect until container rebuild.

### 9.2 Build Time Issues

**Issue:** Docker builds are slow and not optimized for development.

**Problems:**
- **No layer caching** for dependencies
- **Reinstalls everything** on any file change
- **No dev vs prod Dockerfile** - same heavyweight build for both
- **No hot reload** - need full restart for code changes

### 9.3 Health Check Problems

**Issue:** Health endpoint too simplistic.

```python
@application.get("/api/health")
async def health_check():
    return {"status": "ok", "version": __version__}
```

**Missing:**
- **Database connectivity** check
- **Strategy loading** verification
- **Background task** status
- **Memory/CPU** health
- **Data freshness** checks

## 10. Missing Features

### 10.1 Advanced Backtesting Methods

**Critical missing features:**
- **Walk-forward optimization** - test strategies on rolling windows
- **Monte Carlo analysis** - test robustness with randomized data
- **Bootstrap sampling** - statistical confidence intervals
- **Cross-validation** - out-of-sample testing
- **Parameter sensitivity analysis** - how robust are optimal parameters?

### 10.2 Portfolio Management

**Missing:**
- **Multi-strategy portfolios** - combine strategies with allocation rules
- **Correlation analysis** - diversification metrics
- **Risk budgeting** - allocate risk across strategies/sectors
- **Rebalancing** - periodic portfolio adjustments
- **Drawdown-based position sizing** - reduce size after losses

### 10.3 Real-Time Integration

**Missing:**
- **Live trading interface** - connect to brokers for real trading
- **Signal generation API** - real-time signal updates
- **Portfolio monitoring** - live P&L tracking
- **Alert system** - notifications when conditions met

### 10.4 Advanced Analytics

**Missing:**
- **Regime detection integration** - exists but poorly integrated
- **Market microstructure** - bid/ask spreads, order book depth
- **Alternative data** - sentiment, news, fundamental data
- **Machine learning** - pattern recognition, feature engineering

## 11. Specific Bug Analysis

### 11.1 "No Price Data Found" - TV Strategies

**Symptoms:** 2 of 5 TradingView-converted strategies fail with this error despite data existing.

**Suspected causes:**
1. **Symbol name mismatch** - TV uses different ticker format
2. **Date range issues** - strategies request data outside available range  
3. **Timeframe mismatch** - requesting intraday data that doesn't exist
4. **Query logic bug** - SQL query has wrong WHERE conditions

**Investigation needed:**
- Compare symbol names between strategy configs and database
- Check date ranges in failing strategy configs
- Add detailed logging to `_load_*_prices()` methods

### 11.2 tv_atr_trailing_stop Zero Trades

**Symptoms:** Strategy runs without error but generates 0 trades.

**Suspected causes:**
1. **Signal logic bug** - conditions never met due to logic error
2. **Position sizing issue** - calculated position size is 0 or negative
3. **Date alignment** - signals and price data have misaligned dates
4. **Trailing stop bug** - stop logic prevents any entries

**Evidence:** ATR trailing stop should generate many trades if working correctly.

### 11.3 Intraday Strategy Failures

**Critical finding:** ALL intraday backtests lost money.

**Possible framework issues:**
1. **Look-ahead bias** - intraday strategies using future bars
2. **Commission model wrong** - too high for intraday trading frequency
3. **Slippage model inappropriate** - intraday needs different model
4. **Position sizing bug** - fractional shares or wrong calculations
5. **Data quality** - intraday data has more gaps/errors than daily

**This suggests fundamental issues with intraday framework, not just bad strategies.**

### 11.4 Long+Short Systematic Failure

**All 24 strategies became unprofitable** when enabling short side.

**Suspected framework bugs:**
1. **Short position simulation wrong** - P&L calculation error
2. **Signal inversion bug** - buy signals becoming bad sell signals instead of good short signals
3. **Cost model inappropriate** - doesn't account for short selling costs
4. **Portfolio tracking bug** - short positions tracked incorrectly

**Critical:** This suggests the short-selling simulation is fundamentally broken.

## 12. Lessons Learned from Usage

### 12.1 Configuration Drift is Real

**Case study:** kell_base_breakout degraded from PF 2.92 → 0.81

**Potential causes:**
- Config parameters changed without documentation
- Strategy implementation changed but version not bumped
- Data changes (stock splits, adjustments) affected historical results
- Bug introduced in indicator calculations

**Lesson:** Need robust config management and change detection.

### 12.2 Strategy Conversion Has High Error Rate

**Finding:** 69 TV strategies converted, many likely have bugs.

**Conversion challenges:**
- **Pine Script → Python** semantic differences
- **Look-ahead bias** easier to introduce in conversion
- **Parameter mapping** - Pine script defaults may not translate
- **Market simulation differences** - TV vs our engine behavior

**Lesson:** Need better conversion validation and testing process.

### 12.3 Silent Failures Are Common

**Pattern:** Many issues only discovered through manual inspection of results.

**Examples:**
- Strategies generating 0 trades
- Unrealistic results that aren't flagged
- Look-ahead bias not automatically detected
- Configuration errors that don't prevent execution

**Lesson:** Need extensive automated validation and sanity checking.

### 12.4 Debugging Complex Strategies is Hard

**Finding:** When strategies misbehave, very difficult to identify root cause.

**Current process:**
1. Re-run with additional logging
2. Manually inspect indicator values
3. Check signal generation logic
4. Verify trade execution step-by-step

**Problems:** No tooling to automate this debugging process.

## 13. Data Quality Issues

### 13.1 Missing Data Handling

**Issue:** Framework doesn't gracefully handle missing data.

**Scenarios:**
- **Stock delistings** - symbol disappears mid-backtest
- **Trading halts** - no bars for extended periods  
- **Holiday gaps** - missing days not properly handled
- **Corporate actions** - mergers, spinoffs break continuity

### 13.2 Data Validation Gaps

**Missing validations:**
- **Price sanity** - negative prices, impossible high/low relationships
- **Volume validation** - zero volume on active trading days
- **Adjusted close consistency** - verify splits/dividends properly applied
- **Data freshness** - detect stale data that affects recent results

### 13.3 Insufficient Data Preprocessing

**Issue:** Raw data used without cleaning or normalization.

**Missing:**
- **Outlier detection** and correction
- **Data smoothing** for noisy intraday data  
- **Survivorship bias** correction
- **Point-in-time data** - using today's S&P 500 list for historical backtests

## 14. Infrastructure Issues

### 14.1 Database Schema Limitations

**Issues identified:**
- **No audit trail** - can't track config/data changes over time
- **Poor partitioning** - TimescaleDB not optimally configured
- **Missing indexes** - queries slow for large datasets
- **No data compression** - storing more data than necessary

### 14.2 Monitoring/Observability Gaps

**Missing:**
- **Performance metrics** - query times, memory usage, CPU utilization
- **Business metrics** - success rates, avg backtest duration
- **Error aggregation** - patterns in failures
- **Resource usage tracking** - detect memory leaks, runaway processes

### 14.3 Security Issues

**Identified risks:**
- **API wide open** - no authentication on sensitive endpoints
- **SQL injection potential** - if dynamic queries added
- **No rate limiting** - API can be overwhelmed
- **Secrets in logs** - database URLs, API keys may leak

## 15. Recommendations for V2

### 15.1 Architecture Redesign

**Proposed:** Microservice/plugin architecture with clear boundaries:
- **Data Service** - unified data loading, validation, caching
- **Strategy Engine** - isolated strategy execution with lifecycle hooks
- **Portfolio Manager** - realistic position tracking and risk management
- **Execution Simulator** - sophisticated fill simulation
- **Results Service** - metrics, reporting, comparison tools

### 15.2 Better Strategy Framework

**Proposed:** Event-driven strategy pattern:
- **Lifecycle hooks**: `on_start()`, `on_bar()`, `on_fill()`, `on_end()`
- **Stateful strategies** - maintain internal state between bars
- **Portfolio awareness** - access to current positions and cash
- **Dynamic exits** - adjust stops/targets based on current conditions

### 15.3 Robust Data Layer

**Proposed:**
- **Unified data interface** - same API for daily/intraday/alternative data
- **Built-in validation** - automatic data quality checks
- **Smart caching** - intelligent cache invalidation
- **Data versioning** - track data changes over time

### 15.4 Production-Ready Infrastructure

**Proposed:**
- **Proper async task queue** (Celery/RQ) instead of asyncio.create_task()
- **Database optimization** - proper indexes, query optimization
- **Monitoring stack** - Prometheus + Grafana for observability
- **Authentication** - secure API endpoints
- **CI/CD pipeline** - automated testing and deployment

## 16. Priority Issues for V2

### P0 - Framework Bugs (Fix First)
1. **Long+short mode systematic failure** - debug and fix or remove
2. **"No price data found" errors** - fix data loading reliability  
3. **tv_atr_trailing_stop zero trades** - fix signal generation logic
4. **Look-ahead bias** - implement comprehensive bias checking

### P1 - Architecture (Major Refactor)
1. **Modular architecture** - separate data/strategy/execution/results
2. **Proper async task system** - replace background task hack
3. **Strategy lifecycle model** - support stateful strategies
4. **Portfolio simulation accuracy** - realistic fill modeling

### P2 - Developer Experience (Quality of Life)
1. **Strategy debugging tools** - visualization, step-through debugging
2. **Configuration management** - versioning, validation, templates
3. **Comprehensive testing** - unit tests for all strategies
4. **Performance monitoring** - identify bottlenecks automatically

### P3 - Advanced Features (Nice to Have)
1. **Walk-forward optimization** 
2. **Monte Carlo analysis**
3. **Real-time integration**
4. **Alternative data sources**

---

## Conclusion

The trading backtester v1 demonstrates the concept but has fundamental flaws that make it unsuitable for serious trading research. The systematic failure of long+short mode, silent failures, and look-ahead bias issues indicate deep architectural problems that require a complete redesign rather than incremental fixes.

V2 must prioritize correctness, debuggability, and modularity over feature completeness. A smaller, more robust system will be more valuable than a large, unreliable one.

**Next steps:** Use this analysis to design v2 architecture that addresses each identified shortcoming systematically.