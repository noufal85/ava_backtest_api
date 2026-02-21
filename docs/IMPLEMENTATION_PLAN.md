# ava_backtest — Implementation Plan
**Date:** 2026-02-21  
**Kanban epics:** EP-1 through EP-8 (cards #20–27 on dashboard)  
**Design refs:** DESIGN_REVIEW_V3.md, MULTI_MARKET_DESIGN.md  
**Repos:** trading-backtester_v2 (backend), trading-backtester-ui_v2 (frontend)

---

## Timeline Overview

| Week | Epics | Milestone |
|---|---|---|
| 1 | EP-1, EP-2 (start) | Repo running, DB up, FMP data flowing |
| 2 | EP-2 (finish), EP-3 (start) | India data live, engine scaffold |
| 3 | EP-3 (finish), EP-4 | Full engine + all 8 strategies |
| 4 | EP-5, EP-6 | API live, metrics complete |
| 5 | EP-7 (start) | UI scaffold, market selector, dashboard |
| 6 | EP-7 (finish) | All pages complete, charts working |
| 7 | EP-8, polish | Docker stack, CI, monitors, README |

---

## EP-1: Project Foundation
**Kanban card:** #20 | **Priority:** Critical | **Dependency:** None

### Tasks

#### 1.1 Repo bootstrap
- [ ] Clone `trading-backtester_v2` as the working repo
- [ ] Verify `pyproject.toml` deps — add `nsepy`, `upstox-python-sdk`, `optuna`
- [ ] Create `src/core/markets/__init__.py` + `registry.py` (from MULTI_MARKET_DESIGN.md §1)
- [ ] Create `.env.example` with all required keys:
  ```
  DATABASE_URL=postgresql+asyncpg://ava:ava2026@localhost:5435/ava
  REDIS_URL=redis://localhost:6379
  FMP_API_KEY=
  UPSTOX_API_KEY=
  UPSTOX_API_SECRET=
  ZERODHA_API_KEY=
  ZERODHA_API_SECRET=
  ALPACA_API_KEY=
  ALPACA_SECRET_KEY=
  ```

#### 1.2 Docker Compose
- [ ] Start from existing `docker-compose.yml` (TimescaleDB 2.14.2-pg15 + Redis 7 already defined)
- [ ] Add India provider env vars to `backtester-v2` service (from MULTI_MARKET_DESIGN.md §7)
- [ ] Verify services come up: `docker compose up ava-db redis`
- [ ] Confirm TimescaleDB extension available: `SELECT default_version FROM pg_available_extensions WHERE name='timescaledb'`

#### 1.3 Database schema
- [ ] Run `docs/schema/schema.sql` (existing schema)
- [ ] Run SQL patches from MULTI_MARKET_DESIGN.md §3 (add `market_code` to all tables)
- [ ] Seed default US universes (sp500, sp500_liquid, nasdaq100, russell2000)
- [ ] Seed India universes (nifty50, nifty100, nifty500, nse_large_cap, nse_mid_cap, nse_small_cap)
- [ ] Create Alembic migration file matching all patches

#### 1.4 App config (pydantic-settings)
- [ ] `src/core/config.py` — `Settings` class loading from `.env`:
  ```python
  class Settings(BaseSettings):
      database_url: str
      redis_url: str
      fmp_api_key: str = ""
      upstox_api_key: str = ""
      upstox_api_secret: str = ""
      zerodha_api_key: str = ""
      zerodha_api_secret: str = ""
      log_level: str = "INFO"
  ```

#### 1.5 Logging
- [ ] Configure `structlog` with JSON output in production, coloured in dev
- [ ] Add request ID middleware to FastAPI app

**Definition of done:** `docker compose up` starts cleanly. DB schema applied. `pytest tests/` runs (no tests yet, just confirms imports work).

---

## EP-2: Data Provider Layer
**Kanban card:** #21 | **Priority:** Critical | **Dependency:** EP-1

### Tasks

#### 2.1 Provider ABC
- [ ] `src/core/data/providers/base.py` — `DataProvider` ABC (from MULTI_MARKET_DESIGN.md §2)
- [ ] Standard output schema: `pl.DataFrame` with columns `[ts, open, high, low, close, volume, adj_close]`

#### 2.2 FMP Provider (US)
- [ ] `src/core/data/providers/fmp.py` — update existing to implement `DataProvider` ABC
- [ ] Add `market: MarketCode` param to `fetch_ohlcv` (raises if not `MarketCode.US`)
- [ ] `search_symbols()` → FMP `/search` endpoint
- [ ] `get_universe_symbols()` → FMP `/sp500_constituent`, `/nasdaq_constituent`
- [ ] Rate limiting with `aiolimiter` (FMP: 300 req/min on paid, 10/min free)

#### 2.3 NSEpy Provider (India — daily)
- [ ] `src/core/data/providers/nsepy.py` (from MULTI_MARKET_DESIGN.md §2)
- [ ] Hardcode Nifty universe symbol lists (50, 100, 500)
- [ ] Run NSEpy sync in `asyncio.get_event_loop().run_in_executor(None, ...)`
- [ ] Normalise output columns to standard schema
- [ ] Handle NSE symbol format (`RELIANCE.NS` → strip `.NS` for nsepy call)

#### 2.4 Alpaca Provider (US fallback)
- [ ] `src/core/data/providers/alpaca.py` — implement `DataProvider` ABC
- [ ] Uses Alpaca Data API v2 (`aiohttp` calls)
- [ ] Supports daily + intraday timeframes

#### 2.5 Provider Router
- [ ] `src/core/data/providers/router.py` — `ProviderRouter` (from MULTI_MARKET_DESIGN.md §2)
- [ ] `fetch_ohlcv()` tries providers in `MarketConfig.data_providers` order
- [ ] Automatic failover on exception; logs each failure
- [ ] Singleton instantiated in FastAPI app startup

#### 2.6 File Cache
- [ ] `src/core/data/cache/file_cache.py`
- [ ] Cache key: `{market}_{symbol}_{start}_{end}_{timeframe}.parquet` (Polars Parquet for speed)
- [ ] Cache dir: configurable via `Settings.cache_dir` (default: `./data/cache/`)
- [ ] `get()` / `put()` / `invalidate()` / `is_stale(max_age_hours)` methods
- [ ] TTL: 24h for daily, 4h for intraday

#### 2.7 Redis Indicator Cache
- [ ] `src/core/data/cache/indicator_cache.py`
- [ ] Key: `indicator:{market}:{symbol}:{timeframe}:{indicator_name}:{config_hash}:{data_hash}`
- [ ] Store as MessagePack (fast serialisation) via `redis.asyncio`
- [ ] TTL: 1 week (indicators rarely change once computed)

#### 2.8 Data Validator
- [ ] `src/core/data/validator/validator.py`
- [ ] Checks (from DESIGN_SPEC_V2): OHLC consistency (`high >= low`, `high >= open/close`), zero prices, gap detection (`> settlement_days + 5` business days), minimum bar count (100 bars)
- [ ] India-specific: tick size rounding check (warn if prices aren't multiples of `market.tick_size`)
- [ ] Returns `ValidationResult(errors: list[str], warnings: list[str])`
- [ ] Gate: engine rejects data if `errors` is non-empty

#### 2.9 Universe Management
- [ ] `src/core/data/universe/manager.py`
- [ ] `get_symbols(universe_name, market)` — checks DB first, falls back to provider
- [ ] Refreshes DB universe membership nightly (cron task)
- [ ] Custom universe creation (symbol list + filter rules)

**Integration test:** Fetch 1 year of daily OHLCV for `AAPL` (US/FMP) and `RELIANCE.NS` (India/NSEpy), validate output, cache to Parquet, reload from cache. Confirm timestamps use correct market timezone.

---

## EP-3: Backtest Engine Core
**Kanban card:** #22 | **Priority:** Critical | **Dependency:** EP-2

### Tasks

#### 3.1 DataWindow — temporal enforcement
- [ ] `src/core/execution/data_window.py`
  ```python
  class DataWindow:
      def current_bar(self) -> pl.DataFrame     # bar N only
      def historical(self, n: int) -> pl.DataFrame  # bars 0..N-1
      def indicators(self) -> pl.DataFrame      # all computed up to N
  ```
- [ ] Enforces: `current_bar()` never returns future data. `historical()` excludes current bar.
- [ ] Unit test: attempt to access bar N+1 → raises `TemporalViolationError`

#### 3.2 Middleware pipeline
- [ ] `src/core/execution/pipeline.py` — `BacktestPipeline` class
  ```python
  pipeline = BacktestPipeline([
      IndicatorMiddleware(indicators),
      SignalMiddleware(strategy),
      SizingMiddleware(sizer),
      RiskMiddleware(risk_rules),
      ExecutionMiddleware(fill_model, cost_model),
      AttributionMiddleware(logger),
  ])
  ```
- [ ] Each middleware: `process(bar: Bar, state: EngineState) -> EngineState`
- [ ] Pipeline is composable — middlewares are independent, testable in isolation

#### 3.3 Portfolio & Position state
- [ ] `src/core/execution/portfolio.py` — `Portfolio` class (from DESIGN_REVIEW_V3.md)
- [ ] `Position`: `symbol, quantity, avg_cost, market_value, unrealized_pnl, realized_pnl`
- [ ] `Portfolio.apply_fill(fill)` — updates cash, positions, computes realized P&L on close
- [ ] `Portfolio.update_market_values(prices)` — mark-to-market, records equity history
- [ ] Properties: `equity`, `cash`, `gross_exposure`, `net_exposure`, `leverage`, `heat_pct`

#### 3.4 Market-aware cost model
- [ ] `src/core/execution/costs.py` — add `ZerodhaFlatCommission` + `MarketAwareCostModel` (from MULTI_MARKET_DESIGN.md §5)
- [ ] `MarketAwareCostModel.calculate_total()` returns `{commission, stt, gst, stamp_duty, total}`
- [ ] Unit tests: US trade (zero STT), India trade (check STT = 0.1% of value, GST = 18% of commission)

#### 3.5 Fill simulation
- [ ] `src/core/execution/fills.py` — `RealisticFillSimulator` (from v2 design)
- [ ] Market orders: square root impact model + half-spread slippage
- [ ] Limit orders: fill if price touched, queue position model
- [ ] Partial fills for large orders (>5% of avg daily volume)
- [ ] India: respect `tick_size` rounding on fill price (multiples of ₹0.05)

#### 3.6 Bar processing loop
- [ ] `src/core/execution/engine.py` — `BacktestEngine.run()`
- [ ] Execution model from DESIGN_SPEC_V2 §3 (bar N signal → bar N+1 open fill)
- [ ] `pending_signal` pattern — `{BUY, SELL, None}`
- [ ] Stop-loss check on every bar: `if bar.low <= stop_price: fill at stop_price`
- [ ] Progress callback: emits every `ceil(symbols_total * 0.05)` symbols → WebSocket push

#### 3.7 IS/OOS split
- [ ] Add `run_type: Literal["standard", "is_oos", "walk_forward"]` to `CreateBacktestRequest`
- [ ] For `is_oos`: split date computed as `start + (end-start) * is_pct` (default `is_pct=0.7`)
- [ ] Store `is_start/is_end/oos_start/oos_end` on backtest record
- [ ] Compute separate metrics for IS and OOS periods; store in `metrics_extra` JSONB

#### 3.8 Walk-forward analysis
- [ ] `src/core/analytics/walkforward.py`
- [ ] Rolling window: train on N months, test on M months, step by M months
- [ ] Refit parameters on each training window (calls optimiser)
- [ ] Output: per-window `{is_sharpe, oos_sharpe, degradation_pct, optimal_params}`
- [ ] Degradation score: `(IS_sharpe - OOS_sharpe) / IS_sharpe * 100`

**Integration test:** Run EMA crossover on SPY 2020–2024. Verify: (a) no fills at signal bar, (b) fills at next bar open, (c) equity curve is monotonic or drawdown bounded, (d) P&L matches manual calculation for first 3 trades.

---

## EP-4: Strategy Implementations
**Kanban card:** #23 | **Priority:** High | **Dependency:** EP-3

### Tasks

#### 4.1 BaseStrategy
- [ ] `src/core/strategy/base.py` — `BaseStrategy` ABC + `Signal` dataclass (from v2 example)
- [ ] Interface: `get_indicators()`, `get_warmup_periods()`, `generate(data, indicators) -> Signal`
- [ ] `Signal.action: Literal["buy", "sell", "exit", "hold", "short", "cover"]`
- [ ] `Signal.strength: float` (0–1), `Signal.confidence: float` (0–1), `Signal.metadata: dict`

#### 4.2 Strategies (each has: class + YAML config + unit test)

| # | Strategy | File | Key test |
|---|---|---|---|
| 1 | SMA Crossover | `src/strategies/classic/sma_crossover.py` | Cross fires exactly once, not every bar |
| 2 | RSI Mean Reversion | `src/strategies/classic/rsi_mean_reversion.py` | max_hold_days exit triggers |
| 3 | MACD Crossover | `src/strategies/classic/macd_crossover.py` | fast < slow constraint enforced |
| 4 | Bollinger Bands | `src/strategies/classic/bollinger_bands.py` | `close < lower_band` not intraday low |
| 5 | Momentum Breakout | `src/strategies/classic/momentum_breakout.py` | Channel high is `max(high[-N:])` not close |
| 6 | RSI + Vol Filter | `src/strategies/modern/rsi_vol_filter.py` | No entry when ATR percentile > threshold |
| 7 | Dual Momentum | `src/strategies/modern/dual_momentum.py` | Rebalances only every N bars |
| 8 | Opening Range Breakout | `src/strategies/modern/orb.py` | Hold exactly `hold_bars` bars |

- [ ] Each strategy: YAML config file in `src/strategies/configs/`
- [ ] Each strategy: pytest unit test in `tests/unit/strategies/`
- [ ] Regime middleware: `src/core/strategy/regime.py` — loads from `backtester.regime_history`, blocks signals in disallowed regimes

#### 4.3 Strategy registry
- [ ] `src/core/strategy/registry.py` — maps `strategy_name` → class
- [ ] Auto-discovery of strategy classes via `__init_subclass__` or explicit registration
- [ ] `GET /api/v2/strategies` reads from this registry, not DB (DB stores runs, not class definitions)

**Test:** Run all 8 strategies on a known 1-year dataset. Compare signal counts + final return against expected values stored in `tests/regression/expected_results.json`.

---

## EP-5: Analytics & Optimisation
**Kanban card:** #24 | **Priority:** High | **Dependency:** EP-3, EP-4

### Tasks

#### 5.1 Metrics calculator — 20+ metrics
- [ ] `src/core/analytics/metrics.py`

| Metric group | Metrics |
|---|---|
| **Primary** | Total Return %, CAGR %, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown %, Max Drawdown Days |
| **Secondary** | Volatility (ann.), Beta vs benchmark, Alpha, R², Omega Ratio, Ulcer Index |
| **Trade stats** | Win Rate %, Profit Factor, Avg Trade P&L, Avg Winner, Avg Loser, Best/Worst Trade, Avg Hold Days, Max Consecutive Wins/Losses, MAE, MFE |
| **Benchmark** | Benchmark Return, Benchmark Sharpe, Excess Return (Alpha), Information Ratio |

- [ ] `monthly_returns: dict[str, float]` — `{"2020-01": 2.5, "2020-02": -1.3}`
- [ ] Benchmark: buy-and-hold of equal-weighted universe (or SPY/NIFTY50 ETF proxy)

#### 5.2 Parameter optimisation
- [ ] `src/core/analytics/optimization/optimizer.py`
- [ ] Method: Optuna (Bayesian, default) + grid search option
- [ ] Objective functions: `sharpe_ratio`, `calmar_ratio`, `total_return_pct`, `sortino_ratio`
- [ ] `param_space` spec per strategy in YAML config:
  ```yaml
  optimization:
    enabled: true
    method: bayesian
    objective: sharpe_ratio
    trials: 200
    param_space:
      fast_period: {type: int, low: 5, high: 50}
      slow_period: {type: int, low: 20, high: 200}
  ```
- [ ] Each trial stores in `backtester.optimization_results`
- [ ] **Overfitting score**: `(IS_sharpe - OOS_sharpe) / IS_sharpe` — stored per trial
- [ ] Best params highlighted; warn if overfitting_score > 0.3

#### 5.3 Cross-strategy analytics
- [ ] `src/core/analytics/portfolio.py`
- [ ] Correlation matrix: Pearson correlation of daily returns across strategies
- [ ] Combined equity curve: weighted sum of equity curves
- [ ] Portfolio metrics: combined Sharpe, combined max drawdown, diversification ratio

**Test:** Run optimisation with 10 trials on EMA crossover. Confirm all trials stored in DB. Confirm best params improve on default. Confirm overfitting score computed.

---

## EP-6: FastAPI Layer
**Kanban card:** #25 | **Priority:** High | **Dependency:** EP-3, EP-5

### Tasks

#### 6.1 App scaffold
- [ ] `src/api/v2/app.py` — FastAPI app, CORS, Prometheus metrics middleware, structlog middleware
- [ ] Background task pool: `ProcessPoolExecutor(max_workers=4)` for engine runs
- [ ] Startup: instantiate `ProviderRouter`, `DataCache`, check DB connection
- [ ] Shutdown: graceful — wait for running backtests to complete

#### 6.2 Markets endpoint
- [ ] `GET /api/v2/markets` → `list_markets()` from registry — no DB call

#### 6.3 Backtests endpoints
- [ ] `GET  /api/v2/backtests?market=US&status=completed&strategy=ema_crossover&limit=50&offset=0`
- [ ] `POST /api/v2/backtests` — validate market, universe, strategy; create DB record; queue background task
- [ ] `GET  /api/v2/backtests/{id}` — full detail + results
- [ ] `DELETE /api/v2/backtests/{id}` — cancel if running, delete record
- [ ] `GET  /api/v2/backtests/{id}/trades?direction=long&limit=200`
- [ ] `GET  /api/v2/backtests/{id}/equity-curve?resample=D`
- [ ] All endpoints: `market_code` filter applied at DB query level (never cross-market)

#### 6.4 Strategies endpoints
- [ ] `GET /api/v2/strategies?market=IN` — return strategies with India-compatible universe options
- [ ] `GET /api/v2/strategies/{name}?version=latest` — full detail + parameter schema
- [ ] `POST /api/v2/strategies/{name}/backtest` — shortcut runner

#### 6.5 Universes + Data endpoints
- [ ] `GET /api/v2/universes?market=IN` — returns Nifty universes for India, SP500 etc. for US
- [ ] `GET /api/v2/symbols/search?q=RELI&market=IN` — routed to NSEpy/Upstox
- [ ] `GET /api/v2/data/candles?symbol=RELIANCE.NS&market=IN&timeframe=1d&start=2020-01-01&end=2024-12-31`

#### 6.6 Analytics endpoints
- [ ] `GET /api/v2/analytics/correlation?backtest_ids=...` — requires all IDs same market
- [ ] `GET /api/v2/analytics/portfolio?backtest_ids=...&weights=...`

#### 6.7 WebSocket
- [ ] `WS /ws/backtests/{id}` — push frames per DESIGN_REVIEW_V3.md
  ```json
  {"type":"progress","pct":42.5,"current_symbol":"RELIANCE.NS","symbols_done":21,"symbols_total":50}
  {"type":"completed","backtest_id":"...","sharpe":1.45,"total_return_pct":22.3}
  {"type":"error","message":"..."}
  ```
- [ ] Connection manager: dict of `{backtest_id: list[WebSocket]}`
- [ ] Engine calls `await ws_manager.broadcast(backtest_id, event)` from background task

#### 6.8 Error contract
- [ ] All errors return `{"error": "...", "code": "VALIDATION_ERROR|NOT_FOUND|...", "details": {...}}`
- [ ] 422 for validation, 404 for not found, 409 for conflict, 500 for engine errors

**Test:** Full integration test — create backtest via API, poll status until completed, fetch trades + equity curve, verify market isolation (US backtest not returned in India query).

---

## EP-7: React UI
**Kanban card:** #26 | **Priority:** High | **Dependency:** EP-6

### Tasks

#### 7.1 Scaffold from trading-backtester-ui_v2
- [ ] Clone `trading-backtester-ui_v2` as working frontend repo
- [ ] Replace `recharts` with `lightweight-charts` in `package.json`
- [ ] Install: `npm remove recharts && npm install lightweight-charts`
- [ ] Verify dev server starts: `npm run dev`

#### 7.2 MarketContext + provider
- [ ] `src/contexts/MarketContext.tsx` — exactly as in MULTI_MARKET_DESIGN.md §8c
- [ ] Wrap `<App>` with `<MarketProvider>` in `App.tsx`
- [ ] `getSelectedMarket()` utility used by `api/client.ts`

#### 7.3 API client update
- [ ] `src/api/client.ts` — inject `market` param into all calls (from MULTI_MARKET_DESIGN.md §8b)
- [ ] Add `listMarkets()` function
- [ ] TanStack Query cache keys include `market`: `['backtests', market, filters]`

#### 7.4 Navbar + market selector
- [ ] `src/components/layout/Navbar.tsx` — add market selector dropdown (from MULTI_MARKET_DESIGN.md §8e)
- [ ] Flag emoji map: `{US: '🇺🇸', IN: '🇮🇳'}`
- [ ] On market change: invalidate all TanStack Query caches, navigate to `/`

#### 7.5 Formatters update
- [ ] `src/utils/formatters.ts` — market-aware `formatCurrency(value, currency)` (from MULTI_MARKET_DESIGN.md §8f)
- [ ] `en-IN` locale for INR (auto lakh/crore grouping)

#### 7.6 Dashboard page
- [ ] Metrics row: Total Return, Sharpe, Max DD, Active Backtests (pulled for selected market only)
- [ ] Equity curve chart (lightweight-charts `LineChart`) with benchmark overlay
- [ ] Active backtests with self-subscribing `<BacktestProgressBar backtestId={id}>`
- [ ] Strategy leaderboard sorted by Sharpe (market-scoped)
- [ ] Recent backtests list with status badges

#### 7.7 Strategy Catalog page
- [ ] Grid of `<StrategyCard>` components
- [ ] Search + category filter (trend / mean-reversion / momentum / multi-factor)
- [ ] Each card shows: name, version, tags, best Sharpe across all runs for this market

#### 7.8 Strategy Builder page (3 tabs)
- [ ] **Tab 1 – Parameters:** `<ParameterInput>` rendered from `parameter_schema` JSON
- [ ] **Tab 2 – Backtest Setup:** `<DateRangePicker>`, `<UniverseSelector>` (market-aware), sizing config, risk rules, IS/OOS toggle
- [ ] **Tab 3 – Preview & Run:** `<ConfigPreview>` (YAML), `<CostEstimator>` (estimated trades + commission cost), `<ValidationResults>`, Run button
- [ ] `<SurvivorshipBiasWarning>` auto-shows for known tickers (AAPL, TSLA, NVDA, TCS, RELIANCE)
- [ ] Submits to `POST /api/v2/backtests`, navigates to `/backtests/{id}`

#### 7.9 Results Viewer page
- [ ] Metrics summary cards (all 20+ metrics in collapsible groups)
- [ ] `<EquityCurveChart>` (lightweight-charts):
  - IS region: normal colour
  - OOS region: slightly dimmed with vertical divider line
  - Regime overlay: coloured background bands (bull=green tint, bear=red tint)
  - Benchmark overlay: dashed line
  - Synced crosshair with drawdown chart
- [ ] `<DrawdownChart>` (lightweight-charts, HistogramSeries)
- [ ] `<MonthlyReturnsHeatmap>` — calendar grid, green/red cells
- [ ] `<TradeDistribution>` — histogram of trade P&L
- [ ] `<TradesTable>` — virtualised, sortable, filterable; India rows show STT/GST columns
- [ ] Currency displayed from `backtest.currency`

#### 7.10 Portfolio Analytics page
- [ ] Multi-backtest selector (same market only — cross-market blocked with notice)
- [ ] `<CorrelationMatrix>` — heatmap grid
- [ ] `<CombinedEquityCurve>` — weighted overlay
- [ ] `<RiskBudgetTable>` — strategy weights, vol contribution
- [ ] `<DrawdownComparison>` — overlaid drawdown curves

#### 7.11 Optimisation page (`/optimize/:strategyName`)
- [ ] Parameter range sliders (from `param_space` in YAML)
- [ ] Objective function selector
- [ ] Progress bar (WS-connected)
- [ ] `<OptimizationHeatmap>` — 2D grid for two chosen params, colour = objective value
- [ ] `<OverfittingScoreGauge>` — traffic light (green <10%, yellow 10–30%, red >30%)
- [ ] Results table sorted by objective value

#### 7.12 Walk-Forward page (`/walkforward/:runId`)
- [ ] `<WalkForwardResultsChart>` — per-window IS/OOS Sharpe bars
- [ ] Degradation trend line (should be flat or improving)
- [ ] Best params per window table

#### 7.13 Data Management page (`/data`)
- [ ] Provider status per market (FMP UP/DOWN, NSEpy UP/DOWN)
- [ ] Universe membership view — symbols in each universe
- [ ] Data freshness table (`backtester.data_freshness`)
- [ ] Manual "Refresh Data" trigger per symbol/market

---

## EP-8: Infrastructure & Monitoring
**Kanban card:** #27 | **Priority:** Medium | **Dependency:** EP-6, EP-7

### Tasks

#### 8.1 Alembic migrations
- [ ] `alembic init migrations/`
- [ ] Migration 001: base schema (schema.sql)
- [ ] Migration 002: multi-market patches (MULTI_MARKET_DESIGN.md §3)
- [ ] `alembic upgrade head` runs cleanly from scratch

#### 8.2 GitHub repos
- [ ] Create `ava_backtest_api` repo (backend) — prefix with `ava_`
- [ ] Create `ava_backtest_ui` repo (frontend)
- [ ] Push both with initial commits
- [ ] Add to `MEMORY.md` and dashboard Projects page

#### 8.3 GitHub Actions CI
- [ ] `.github/workflows/ci.yml`:
  ```yaml
  on: [push, pull_request]
  jobs:
    test:
      - ruff check src/
      - mypy src/
      - pytest tests/unit/ -x
    integration:
      - docker compose up -d ava-db redis
      - pytest tests/integration/ -x
  ```

#### 8.4 Uptime Kuma monitors
- [ ] HTTP: `GET /api/v2/markets` — API health (every 60s)
- [ ] Push: backtest worker (pings on task completion)
- [ ] HTTP: `GET /api/v2/data/candles?symbol=AAPL&...` — data freshness
- [ ] HTTP: `GET /api/v2/data/candles?symbol=RELIANCE.NS&market=IN&...` — India data freshness
- [ ] All tagged `ava-backtest`

#### 8.5 README
- [ ] Quickstart: `cp .env.example .env`, fill keys, `docker compose up`, open `http://localhost:5173`
- [ ] Architecture diagram (Excalidraw + Mermaid fallback)
- [ ] Strategy list with brief descriptions
- [ ] How to add a new market (4-step playbook)
- [ ] How to add a new strategy (class + YAML + test)

---

## Dependencies Graph

```
EP-1 (Foundation)
  └─→ EP-2 (Data Layer)
        └─→ EP-3 (Engine Core)
              ├─→ EP-4 (Strategies)
              └─→ EP-5 (Analytics)
                    └─→ EP-6 (API)
                          └─→ EP-7 (UI)
                                └─→ EP-8 (Infra)
```

EP-4 and EP-5 can run in parallel after EP-3.  
EP-8 can start in parallel with EP-7 (infra work is independent of UI).

---

## Definition of Done (per epic)

| Epic | DoD |
|---|---|
| EP-1 | `docker compose up` clean, DB schema applied, imports work |
| EP-2 | Fetch AAPL + RELIANCE.NS, validate, cache, reload from cache — all pass |
| EP-3 | EMA crossover on SPY 2020–2024: no lookahead, fills at N+1 open, P&L matches manual calc |
| EP-4 | All 8 strategies produce signals on test data; regression results match expected |
| EP-5 | 20 metrics computed, Optuna runs 10 trials, overfitting score stored |
| EP-6 | API integration test: create→run→fetch results for both US and India backtest |
| EP-7 | All pages render, market selector switches US↔IN and data updates, no cross-market leaks |
| EP-8 | `alembic upgrade head` from scratch, CI passes, Uptime Kuma monitors green |

---

## Estimated Effort

| Epic | Estimated Days | Notes |
|---|---|---|
| EP-1 | 1 | Mostly config + existing schema |
| EP-2 | 3 | FMP existing, NSEpy new, router + cache new |
| EP-3 | 5 | Engine is the hardest and most critical piece |
| EP-4 | 3 | 8 strategies × ~2h each + tests |
| EP-5 | 3 | Metrics calc + Optuna integration |
| EP-6 | 3 | FastAPI endpoints + WS |
| EP-7 | 7 | Most pages + chart migration from Recharts → lightweight-charts |
| EP-8 | 2 | Alembic + CI + monitors |
| **Total** | **27 days** | ~5–6 weeks at comfortable pace |
