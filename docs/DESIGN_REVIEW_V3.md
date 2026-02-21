# ava_backtest — Design Review V3
**Reviewed:** 2026-02-20  
**Sources:** trading-backtester_v2, trading-backtester-ui_v2, ava_backtest DESIGN_SPEC_V2  
**Reviewer:** Ava (cross-design synthesis)

---

## Executive Summary

The v2 repos provide a significantly more production-grade architecture than our current ava_backtest design. Rather than choosing one over the other, we should **merge the best of both**:

- **From trading-backtester_v2**: Polars, TimescaleDB, middleware pipeline, walk-forward analysis, parameter optimization, multi-timeframe, indicator caching, strategy versioning, WebSocket real-time updates
- **From ava_backtest DESIGN_SPEC_V2**: 8 concrete strategy definitions, regime detection, IS/OOS split, lightweight-charts (better than Recharts), detailed 20+ metrics spec, proven execution model, data validation gate
- **New additions from cross-review**: Pine Script converter UI, universe management page, optimization overfitting detector, walk-forward visualizer, multi-strategy correlation dashboard

---

## 1. Backend Architecture Review

### ✅ Strengths of trading-backtester_v2

| Feature | Why It's Better |
|---|---|
| **Polars data processing** | 10–100x faster than Pandas on large symbol sets. Essential for multi-symbol optimization runs |
| **TimescaleDB hypertables** | Compression, partitioning, and time-series queries miles ahead of SQLite at scale. Symbol+timeframe partitioning is smart |
| **Middleware pipeline** | Clean separation of concerns. `IndicatorMiddleware → SignalMiddleware → SizingMiddleware → RiskMiddleware → ExecutionMiddleware` is more composable than monolithic engine |
| **Plugin/Builder architecture** | `StrategyBuilder().with_signals().with_sizing().with_risk().build()` enables mixing and matching components without code changes |
| **DataWindow temporal enforcement** | Enforces look-ahead prevention at the API level, not just by convention. `window.current()` vs `window.historical()` is an elegant solution |
| **Walk-forward analysis** | IS/OOS split is a good start, but true walk-forward (rolling windows) is far more robust. The schema with `walkforward_results` and degradation_pct is excellent |
| **Parameter optimization** | Optimization trials with `overfitting_score` is critical for preventing curve-fitting. ava_backtest has zero optimization support |
| **Indicator caching with hash keys** | Caching by `config_hash + data_hash` means never recomputing the same indicator twice. Huge speedup for optimization runs |
| **Strategy versioning** | Semantic versioning + code hash means every backtest is reproducible indefinitely |
| **WebSocket real-time progress** | Polling is clunky; WS progress updates (every 5%) make the UX feel live. Essential for 10-minute optimization runs |
| **Multi-timeframe native** | `MultiTimeframeData` with forward-fill alignment is the correct approach. ava_backtest has no multi-timeframe support |
| **Short-selling support** | `ShortSellingCostModel` with borrow rate tiers and availability modeling is a major feature gap in ava_backtest |
| **Venue models** | NASDAQ/NYSE/BATS/IEX microstructure differences in fill simulation. Adds realism |

### ⚠️ Gaps in trading-backtester_v2 vs ava_backtest

| Gap | Impact | Fix |
|---|---|---|
| **No concrete strategy implementations** | The architecture is excellent but has zero strategies. ava_backtest has 8 precisely specified strategies with known-risks documented | Copy all 8 strategies from ava_backtest DESIGN_SPEC_V2 as the initial strategy library |
| **No regime detection** | ava_backtest's regime-aware trading (bull/bear/neutral filter) is a novel differentiator. Strategy 6 (RSI + Vol Filter) is a lightweight version of this | Port regime detection module and integrate as a first-class `RegimeMiddleware` in the pipeline |
| **No data validation gate** | v2 mentions "data quality checks" in the module layout but provides no spec. ava_backtest has explicit validation: OHLC consistency, zero prices, gap detection, min bar count | Implement the ava_backtest validation spec as the `DataValidator` component |
| **No survivorship bias warning** | ava_backtest has a prominent UI warning for single-ticker tests. v2 has nothing. This is UX-critical for honest backtesting | Add as a backend flag in the results response and surface prominently in UI |
| **Vague commission model** | v2 spec shows `InteractiveBrokersCommission` with tier structure, but ava_backtest specifies the exact global params (`commission_per_share`, `slippage_pct`) that appear in the UI | Use ava_backtest's parameter approach for simplicity; add IBKR-tier model as an advanced option |
| **No IS/OOS split** | ava_backtest has explicit IS/OOS configuration with separate metrics display and visual chart separation | Add as a `BacktestMode` enum: `standard | is_oos | walk_forward` |
| **No 20+ metrics spec** | v2 mentions metrics but doesn't define them. ava_backtest has a complete spec: 20+ metrics in Primary, Secondary, Trade Stats, Benchmark categories | Import ava_backtest's metrics specification wholesale |
| **SQLite → TimescaleDB jump is too big for v1** | TimescaleDB requires a PostgreSQL server. For initial deployment this is heavyweight | Start with SQLite (ava_backtest approach), design schema to be TimescaleDB-compatible for future migration. Use UUID PKs and TIMESTAMPTZ types |

### 🆕 New Additions From Cross-Review

| Addition | Rationale |
|---|---|
| **Pine Script converter UI** | v2 mentions a `converter/` module. Surface this in the UI as a tab in Strategy Builder — paste Pine Script, get Python strategy skeleton. Even 60% accurate conversion saves hours |
| **Unusual options activity signal** | Learned from ava_crowd_trader design: UOA is a strong catalyst signal. Add as an optional enrichment signal to any strategy |
| **Optimization overfitting detector** | v2's `overfitting_score` in optimization trials. Surface this as a traffic-light (green/yellow/red) warning in the UI when best params are suspiciously good |
| **Strategy correlation tracker** | When running multiple strategies, show cross-strategy return correlation. Strategies with >0.7 correlation don't diversify. Integrate into Portfolio Analytics |
| **Market regime classifier** | Port from ava_crowd_trader: news-driven regime detection (not just price-based). This would be a `NewsRegimeMiddleware` as an optional pipeline component |

---

## 2. UI/UX Architecture Review

### ✅ Strengths of trading-backtester-ui_v2

| Feature | Why It's Better |
|---|---|
| **More complete page structure** | 6 routes vs ava_backtest's 3 pages. `/data` (universe management), `/analytics` (portfolio), `/settings` are essential for a real platform |
| **WebSocket provider as context** | Global WS connection with React Context is cleaner than per-component connections |
| **TanStack Query (React Query)** | Better server state management than manual fetch + useState patterns |
| **ParameterPresets component** | Named presets per strategy (e.g., "Conservative", "Aggressive") is great UX — saves users from manually re-entering known-good params |
| **ComparisonView component** | Side-by-side comparison of 2–4 backtests. ava_backtest had this in the spec but no component design |
| **UniverseSelector component** | Explicit universe management UI (SP500, SP500_liquid, custom). ava_backtest has no universe concept |
| **BacktestProgressBar as self-subscribing** | Component subscribes to WS internally using backtestId — elegant isolation |
| **MonthlyReturnsHeatmap** | Calendar heatmap of monthly returns is a standard quantitative analysis tool. Missing from ava_backtest |
| **TradeDistribution chart** | Histogram of trade P&L distribution. Critical for understanding whether edge comes from a few outliers |
| **Formik + Yup form management** | Better than uncontrolled forms for complex parameter validation |

### ⚠️ Gaps in trading-backtester-ui_v2

| Gap | Fix |
|---|---|
| **Chart library: Chart.js / Recharts** | ava_backtest correctly chose `lightweight-charts` (TradingView). Recharts struggles with large datasets and has no native crosshair sync. Use lightweight-charts for ALL charts |
| **No IS/OOS visual separation** | ava_backtest's approach of visually separating in-sample vs out-of-sample on the equity curve chart with a vertical divider is critical for understanding result validity |
| **No survivorship bias warning** | Missing entirely. Add a prominent warning banner when single-ticker backtests are run on well-known tickers (AAPL, TSLA, NVDA, etc.) |
| **No walk-forward visualization** | v2's backend supports walk-forward analysis but the UI has no page/component for it. Need a dedicated `WalkForwardResults` component showing: rolling window performance, IS vs OOS Sharpe per window, degradation chart |
| **No optimization UI** | Backend has full optimization support but UI has no parameter sweep page. Need: parameter range configuration, objective function selector (maximize Sharpe/Calmar), progress tracking, results table with overfitting score |
| **No Pine Script import** | Backend has a converter module. Need a `PineScriptImporter` component in Strategy Builder — text area for pasting Pine Script, "Convert" button, diff view of generated Python |
| **Settings page is vague** | "User preferences, API keys" — needs to specify: data provider API key management, default commission/slippage presets, theme (already dark-only is fine), notification preferences |
| **No backtest queue / rate limiting** | If 10 optimization trials run simultaneously, the UI has no queue concept. Need a job queue visualization |
| **StrategyBuilder Preview tab lacks validation feedback** | `ValidationResults` shows errors/warnings but doesn't show a realistic cost estimate (e.g., "estimated 847 trades, ~$423 in commission at current settings") |
| **No mobile-responsive design** | Dark trading dashboard should at minimum be usable on a tablet for monitoring active runs |

### 🆕 New UI Components to Add

| Component | Purpose |
|---|---|
| `WalkForwardResultsChart` | Rolling window IS/OOS Sharpe comparison, degradation trend line |
| `OptimizationHeatmap` | 2D heatmap for two-parameter sweeps (e.g., fast_period vs slow_period, color = Sharpe) |
| `OverfittingScoreGauge` | Traffic-light gauge (green/yellow/red) showing overfitting risk |
| `PineScriptImporter` | Paste Pine Script → convert to Python strategy skeleton |
| `CostEstimator` | Before running: estimate trade count, total commission, effective slippage drag |
| `RegimeOverlay` | On equity curve: color-code background by detected market regime (bull=green, bear=red, neutral=gray) |
| `UniverseBuilder` | Custom universe creation: sector filter, market cap filter, liquidity filter, saved as named universes |
| `BacktestQueue` | Job queue visualization: pending, running (with progress), completed, failed |

---

## 3. Merged Implementation Plan

### Phase 1 — Foundation (Week 1–2)

**Backend:**
- [ ] Project scaffold: FastAPI + SQLite (WAL mode) + Polars + Pydantic V2 + Redis (optional, for indicator cache)
- [ ] Data layer: FMP provider + file cache (ava_backtest approach) → migrate to indicator DB cache later
- [ ] Data validation gate (ava_backtest spec: OHLC consistency, zero prices, gap detection, min bar count)
- [ ] TimescaleDB-ready schema (UUID PKs, proper types) even if running on SQLite initially

**Engine:**
- [ ] DataWindow class (v2's temporal enforcement)
- [ ] Middleware pipeline: Indicator → Signal → Sizing → Risk → Execution → Attribution
- [ ] ava_backtest execution model (pending_signal pattern, bar N+1 open fills)
- [ ] Commission + slippage models (both global-param and IBKR-tier)

**Strategies:**
- [ ] Port all 8 strategies from ava_backtest DESIGN_SPEC_V2 as initial implementations

---

### Phase 2 — Core Analytics (Week 3)

- [ ] All 20+ metrics from ava_backtest spec
- [ ] IS/OOS split mode
- [ ] Basic walk-forward analysis (rolling windows)
- [ ] Short selling support (ShortSellingCostModel from v2)
- [ ] Regime detection middleware (bull/bear/neutral)
- [ ] WebSocket progress events (every 5%)

---

### Phase 3 — Optimization & Advanced Features (Week 4)

- [ ] Parameter optimization framework (Optuna or grid search)
- [ ] Overfitting score calculation
- [ ] Strategy versioning (semantic version + code hash)
- [ ] Indicator cache (hash-keyed, major speedup for optimization runs)
- [ ] Multi-timeframe native support (MultiTimeframeData with forward-fill alignment)
- [ ] Pine Script → Python converter (skeleton generator)

---

### Phase 4 — UI (Week 5–6)

**Using:** Next.js 14 App Router + TypeScript + TailwindCSS + shadcn/ui + **lightweight-charts** + TanStack Query + Formik

**Pages:**
- [ ] `/` — Dashboard (v2 layout with leaderboard, active backtests, equity curve)
- [ ] `/strategies` — Strategy Catalog (grid, search, filter by category)
- [ ] `/strategies/:name` — Strategy Builder (Parameters → Backtest Setup → Preview & Run tabs)
- [ ] `/backtests` — Backtest Queue (running, pending, history)
- [ ] `/backtests/:runId` — Results Viewer (equity curve + IS/OOS divider, drawdown, monthly heatmap, trade distribution, trades table)
- [ ] `/optimize/:strategyName` — Optimization (parameter ranges, objective, heatmap results, overfitting gauge)
- [ ] `/walkforward/:runId` — Walk-Forward Results
- [ ] `/analytics` — Portfolio Analytics (strategy correlation matrix, combined equity curve)
- [ ] `/data` — Universe & Data Management
- [ ] `/settings` — API keys, commission defaults, preferences

**Key components:**
- [ ] `BacktestProgressBar` (self-subscribing WS)
- [ ] `EquityCurveChart` with IS/OOS divider + regime overlay (lightweight-charts)
- [ ] `MonthlyReturnsHeatmap`
- [ ] `TradeDistribution`
- [ ] `OptimizationHeatmap` (2D parameter sweep)
- [ ] `OverfittingScoreGauge`
- [ ] `WalkForwardResultsChart`
- [ ] `PineScriptImporter`
- [ ] `CostEstimator` (pre-run cost projection)
- [ ] `SurvivorshipBiasWarning` (auto-triggers for known-name tickers)

---

## 4. Priority Rankings (Top 10 Build-First)

| # | Item | Why First |
|---|---|---|
| 1 | **DataWindow + temporal enforcement** | Without this, every backtest result is suspect. Foundation of everything |
| 2 | **Middleware pipeline scaffold** | All other engine components plug into this. Can't add strategies without it |
| 3 | **8 strategies from ava_backtest** | Concrete, well-specified, cover trend/momentum/mean-reversion/regime. Immediate usability |
| 4 | **Data validation gate** | Garbage in = garbage out. Block invalid data before it reaches the engine |
| 5 | **20+ metrics spec** | Results without comprehensive metrics are just numbers. Users need Sharpe, Sortino, Calmar, MAE/MFE, etc. |
| 6 | **IS/OOS split** | Prevents users from showing live results and calling them validated |
| 7 | **WebSocket progress + BacktestProgressBar** | Long runs without feedback feel broken. Users will think it crashed |
| 8 | **Walk-forward analysis** | The single most important tool for validating strategy robustness |
| 9 | **Parameter optimization + overfitting score** | Massive value-add; prevents the most common quantitative mistake |
| 10 | **Regime detection middleware** | Differentiator from all public backtesting tools. Strategy 6 (RSI+Vol) and Dual Momentum already hint at regime awareness |

---

## 5. Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Database | **SQLite → TimescaleDB migration path** | SQLite for initial build. Schema uses UUID + TIMESTAMPTZ. TimescaleDB when data exceeds 5GB or multi-user |
| Data processing | **Polars (primary) + Pandas (compat)** | Polars for engine computation, Pandas for existing indicators that use it. Gradual migration |
| Charts | **lightweight-charts for all** | Synced crosshairs, candlestick native, lightweight, TradingView standard. Recharts is not fit for trading |
| UI structure | **Next.js 14 App Router** | Same as ava_backtest, proven with the dashboard |
| Routing | **v2's 10-page structure** | More comprehensive than ava_backtest's 3 pages; needed for a full-featured platform |
| Execution model | **ava_backtest's explicit bar-loop** | Well-specified, understood, easier to test than v2's abstract model |
| Commission params | **Global params in run config** | Simpler UX than per-strategy commissions; IBKR-tier model available as advanced option |

---

*This review supersedes ava_backtest DESIGN_REVIEW.md. All build work should proceed from this document.*
