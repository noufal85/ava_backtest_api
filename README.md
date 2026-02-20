# Trading Backtester V2

**A professional quantitative trading research platform — ground-up rewrite for speed, composability, and correctness.**

> ⚠️ **This repo contains the complete design documentation for V2.** Implementation follows the phased plan in [`docs/design/IMPLEMENTATION_PLAN.md`](docs/design/IMPLEMENTATION_PLAN.md).

## Why V2?

V1 of the trading backtester grew to 95+ strategies with a FastAPI backend, TimescaleDB, and a React UI. It worked — but it hit fundamental architectural limits. After a thorough analysis ([see V1 shortcomings](docs/analysis/v1_shortcomings.md)), we identified **16 categories of issues** including:

- **Look-ahead bias** present in multiple strategies
- **Monolithic engine** that made debugging impossible
- **Long+short mode systematically destroyed returns** for ALL strategies
- **Silent failures** — strategies generating 0 trades with no errors
- **Poor fill simulation** — all trades at close price with fixed slippage
- **No walk-forward optimization** or Monte Carlo analysis
- **Configuration drift** — strategy performance degrading without detection

**V2 addresses every one of these issues** through a composable, event-driven architecture that prevents look-ahead bias by construction.

## V2 Vision

Transform the backtester from a testing tool into a **production-grade quantitative research platform** that:

1. **Prevents look-ahead bias by construction** — temporal data windows enforce this at the API level
2. **Makes strategies composable** — mix and match indicators, sizing, risk rules like building blocks
3. **Runs 10x faster** — Polars + parallel execution + smart caching
4. **Is deeply testable** — every component can be unit tested in isolation
5. **Supports advanced analysis** — walk-forward optimization, Monte Carlo simulation, parameter sensitivity

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.11+ | Core platform |
| **API** | FastAPI | Async REST + WebSocket |
| **Data Processing** | Polars (primary), Pandas (compat) | 10-100x faster DataFrames |
| **Database** | TimescaleDB (PostgreSQL 15) | Time-series optimized storage |
| **Cache** | Redis | L2 cache for indicators and data |
| **Validation** | Pydantic V2 | Config and data validation |
| **Optimization** | scikit-optimize / Optuna | Bayesian parameter optimization |
| **Monitoring** | Prometheus + Grafana | Observability |
| **Linting** | Ruff | Fast Python linting + formatting |
| **Testing** | pytest + pytest-xdist | Parallel test execution |
| **Frontend** | React + TypeScript + Vite | [Separate repo](https://github.com/noufal85/trading-backtester-ui_v2) |

## Design Documents

These docs are the blueprint for V2. Read them in this order:

### 1. [V1 Shortcomings Analysis](docs/analysis/v1_shortcomings.md)
What went wrong in V1 — 16 categories of issues across architecture, data, strategy pattern, execution, API/UI, testing, performance, configuration, deployment, and missing features. This is the "why" behind every V2 design decision.

### 2. [Architecture](docs/design/ARCHITECTURE.md)
The big picture — module layout, data flow, dependency graph, technology choices, database schema, performance targets, and migration strategy. Start here for the overall vision.

### 3. [Strategy Pattern](docs/design/STRATEGY_PATTERN.md)
The composable strategy framework — `BaseStrategy` interface, pluggable indicators, signal generators, position sizers, risk managers. Includes Pine Script conversion framework, multi-timeframe support, and the strategy builder pattern.

### 4. [Data Layer](docs/design/DATA_LAYER.md)
Pluggable data providers, 3-tier caching (Memory → Redis → TimescaleDB), data validation pipeline, universe management, missing data handling, corporate actions, and real-time data integration.

### 5. [Execution Engine](docs/design/EXECUTION_ENGINE.md)
Realistic fill simulation with market microstructure, commission models (IBKR, zero-commission), portfolio state management, walk-forward testing, Monte Carlo analysis, and parallel processing.

### 6. [API & UI](docs/design/API_AND_UI.md)
FastAPI endpoints, WebSocket real-time updates, background task processing, React dashboard design, dark theme design system, and error handling.

### 7. [Implementation Plan](docs/design/IMPLEMENTATION_PLAN.md)
5-phase build plan over ~20 weeks with detailed code examples, testing strategy, migration runbook, risk assessment, and success criteria for each phase.

### Diagrams
- [`docs/diagrams/system_architecture.excalidraw`](docs/diagrams/system_architecture.excalidraw)
- [`docs/diagrams/module_layout.excalidraw`](docs/diagrams/module_layout.excalidraw)
- [`docs/diagrams/data_flow.excalidraw`](docs/diagrams/data_flow.excalidraw)
- [`docs/diagrams/strategy_lifecycle.excalidraw`](docs/diagrams/strategy_lifecycle.excalidraw)

## Repository Structure

```
trading-backtester_v2/
├── README.md                    # You are here
├── docs/
│   ├── analysis/
│   │   └── v1_shortcomings.md   # V1 post-mortem
│   ├── design/
│   │   ├── ARCHITECTURE.md      # System architecture
│   │   ├── STRATEGY_PATTERN.md  # Strategy framework
│   │   ├── DATA_LAYER.md        # Data infrastructure
│   │   ├── EXECUTION_ENGINE.md  # Backtest engine
│   │   ├── API_AND_UI.md        # API + frontend design
│   │   └── IMPLEMENTATION_PLAN.md # Build roadmap
│   └── diagrams/                # Excalidraw diagrams
├── src/
│   ├── core/
│   │   ├── data/                # Data layer (providers, cache, validation)
│   │   ├── strategy/            # Strategy framework (base, registry)
│   │   ├── execution/           # Backtest engine (fills, portfolio)
│   │   ├── sizing/              # Position sizing (kelly, fixed, vol-scaled)
│   │   ├── risk/                # Risk management middleware
│   │   ├── analytics/           # Performance metrics + attribution
│   │   └── indicators/          # Technical indicators library
│   ├── strategies/              # Strategy implementations
│   │   ├── classic/             # Traditional strategies
│   │   ├── modern/              # Advanced strategies
│   │   └── experimental/        # Research strategies
│   ├── api/                     # FastAPI application
│   │   ├── v2/                  # V2 endpoints
│   │   ├── ws/                  # WebSocket handlers
│   │   └── middleware/          # CORS, auth, rate limiting
│   └── cli/                     # Command-line interface
├── tests/
│   ├── unit/                    # Component-level tests
│   ├── integration/             # Multi-component tests
│   ├── regression/              # V1 vs V2 parity tests
│   └── benchmarks/              # Performance benchmarks
├── migrations/                  # TimescaleDB schema migrations
├── docker-compose.yml           # Development environment
└── pyproject.toml               # Project config + dependencies
```

## Key Innovations

### 1. Temporal Data Windows (No Look-Ahead by Construction)
```python
class MarketData:
    def current_bar(self) -> pl.DataFrame:
        return self._df[self.current_idx:self.current_idx + 1]

    def historical(self, lookback: int) -> pl.DataFrame:
        return self._df[max(0, self.current_idx - lookback):self.current_idx]
    # Future data is simply inaccessible
```

### 2. Composable Strategy Builder
```python
strategy = StrategyBuilder("my_strategy") \
    .add_indicator(BollingerBands(period=20)) \
    .add_indicator(RSI(period=14)) \
    .position_sizer(KellySizer(max_kelly=0.2)) \
    .risk_manager(RiskManager()
        .add_rule(TrailingStop(pct=0.02))
        .add_rule(MaxExposure(pct=0.8))) \
    .build()
```

### 3. 10-Line Strategy Implementation
```python
@strategy("ema_crossover", "2.0.0")
class EMACrossover(BaseStrategy):
    def __init__(self, fast=9, slow=21):
        self.fast = EMA(period=fast)
        self.slow = EMA(period=slow)

    def get_indicators(self):
        return [self.fast, self.slow]

    def get_signal_generator(self):
        return CrossoverSignalGenerator("ema_fast", "ema_slow")
```

## Performance Targets

| Metric | V1 Baseline | V2 Target |
|--------|------------|-----------|
| Single symbol backtest | ~2-3 seconds | **<500ms** |
| 100 symbols (4 years) | ~5-10 minutes | **<2 minutes** |
| Memory (large backtest) | ~500MB | **<250MB** |
| Parameter optimization (1000 combos) | N/A | **<30 minutes** |

## Getting Started (Developer Guide)

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- Git

### Setup
```bash
# Clone the repo
git clone https://github.com/noufal85/trading-backtester_v2.git
cd trading-backtester_v2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev,optimization]"

# Start infrastructure
docker compose up -d ava-db redis

# Run database migrations
psql -h localhost -p 5435 -U ava -d ava -f migrations/001_v2_schema.sql

# Run tests
pytest tests/unit/ -v

# Start development server
uvicorn src.api.main:app --host 0.0.0.0 --port 8201 --reload
```

### Development Workflow
```bash
# Lint + format
ruff check src/ --fix
ruff format src/

# Run full test suite
pytest tests/ -v --timeout=300

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only

# Type checking
mypy src/
```

## Related Repos

- **Frontend**: [trading-backtester-ui_v2](https://github.com/noufal85/trading-backtester-ui_v2) — React + TypeScript + Vite dashboard
- **V1 Backtester**: Original implementation (being superseded)

## Implementation Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | 📋 Planned | Data Layer + Core Abstractions |
| Phase 2 | 📋 Planned | Strategy Framework + Indicators |
| Phase 3 | 📋 Planned | Execution Engine + Portfolio |
| Phase 4 | 📋 Planned | API + UI + Optimization |
| Phase 5 | 📋 Planned | Migration + Validation + Cutover |

## License

Private — All rights reserved.
