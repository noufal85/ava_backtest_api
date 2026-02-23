# Ava Backtest Platform

Multi-market quantitative trading research platform with 124 strategies, AI-powered strategy builder, and Docker-first deployment.

## Quick Start

```bash
# Clone and configure
git clone https://github.com/noufal85/ava_backtest_api.git
cd ava_backtest_api
cp .env.example .env  # Add your FMP, Alpaca, AlphaVantage, Finnhub keys

# Start everything
docker compose up -d

# Verify
curl http://localhost:8201/api/v2/strategies | python3 -c "import json,sys; print(f'{len(json.load(sys.stdin))} strategies loaded')"
```

- **API**: http://localhost:8201 (FastAPI + Swagger at `/docs`)
- **UI**: http://localhost:8203 (React + TradingView charts)

## Features

| Feature | Details |
|---------|---------|
| **124 Strategies** | 9 built-in + 115 custom (trend, mean reversion, momentum, pattern, volatility, calendar) |
| **AI Strategy Builder** | Convert text descriptions, Pine Script, or screenshots → registered Python strategy |
| **Multi-Market** | USA (NYSE/NASDAQ) + India (NSE/BSE) with extensible market registry |
| **Walk-Forward Analysis** | Out-of-sample validation to detect overfitting |
| **Optuna Optimization** | Bayesian hyperparameter tuning with overfitting score |
| **20+ Analytics Metrics** | Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor |
| **Real-Time WebSocket** | Live progress updates streamed to UI during backtests |
| **TradingView Charts** | Interactive equity curves via lightweight-charts |
| **Docker-First** | Full stack in 4 containers: TimescaleDB, Redis, FastAPI, nginx UI |

## Architecture

```mermaid
graph TD
    UI["React UI (nginx :8203)"] --> API["FastAPI :8201"]
    API --> Engine["BacktestEngine"]
    API --> WS["WebSocket /ws/backtests/{id}"]
    Engine --> Strategies["124 Strategies"]
    Engine --> Providers["Data Providers (FMP, Alpaca)"]
    Engine --> Analytics["Metrics Calculator"]
    API --> DB["TimescaleDB :5435"]
    API --> Cache["Redis :6379"]
    Strategies --> Classic["9 Built-in"]
    Strategies --> Custom["115 Custom (auto-discovered)"]
```

## Stack

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `ava-db` | `timescale/timescaledb:2.14.2-pg15` | 5435 | Market data + results |
| `redis` | `redis:7-alpine` | 6379 | Cache + pub/sub |
| `backtester-v2` | Custom Python 3.11 | 8201 | FastAPI backend |
| `ava-ui` | Custom nginx | 8203 | React frontend |

## API Endpoints

### Strategies
```
GET  /api/v2/strategies                    # List all (124 strategies)
GET  /api/v2/strategies/{name}             # Detail with parameter schema
```

### Backtests
```
POST /api/v2/backtests                     # Create + run backtest
GET  /api/v2/backtests                     # List runs (?market=US&status=completed)
GET  /api/v2/backtests/{id}                # Get result (metrics, trades, equity curve)
GET  /api/v2/backtests/{id}/trades         # Paired trade records
GET  /api/v2/backtests/{id}/equity-curve   # YYYY-MM-DD + equity points
WS   /ws/backtests/{id}                    # Live progress stream
```

### Example: Run a Backtest
```bash
curl -X POST http://localhost:8201/api/v2/backtests \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name": "sma_crossover",
    "universe": "sp500_liquid",
    "start_date": "2023-01-01",
    "end_date": "2024-06-01",
    "initial_capital": 100000,
    "market": "US"
  }'
```

## Strategies

### Built-in (9)
| Strategy | Category | Description |
|----------|----------|-------------|
| `sma_crossover` | Trend | Fast/slow SMA crossover |
| `ema_crossover` | Trend | Fast/slow EMA crossover |
| `macd_crossover` | Trend | MACD line/signal crossover |
| `rsi_mean_reversion` | Mean Reversion | RSI oversold/overbought |
| `bollinger_bands` | Mean Reversion | Bollinger Band bounce |
| `momentum_breakout` | Momentum | Donchian channel breakout |
| `dual_momentum` | Momentum | Absolute + relative momentum |
| `opening_range_breakout` | Momentum | First N-bar range breakout |
| `rsi_vol_filter` | Multi-Factor | RSI with volume confirmation |

### Custom (115) — Migrated from trading-backtester
All auto-discovered from `src/strategies/custom/`. Highlights:

| Strategy | Category | Profit Factor* | Win Rate* |
|----------|----------|---------------|-----------|
| `tv_rsi_ema` | Mean Reversion | 11.27 | 85.2% |
| `tv_rmi_3_strategy` | Momentum | 9.53 | — |
| `dg_ma_cross` | Trend | 3.72 | 41.3% |
| `dg_macd_ema` | Trend | 2.38 | 30.8% |
| `dg_bb_rsi` | Mean Reversion | 2.00 | — |
| `tv_zscore_mean_reversion` | Mean Reversion | 1.69 | 66.0% |
| `connors_rsi` | Mean Reversion | 0.98 | 51.0% |
| `tv_ichimoku_cloud` | Trend | — | — |

*Results from previous backtester — re-run on this platform for updated metrics.

See [docs/strategies/](docs/strategies/) for full catalog.

### Adding a New Strategy

**Option 1: AI Strategy Builder**
```bash
python3 skills/strategy-builder/build_strategy.py \
  --name my_strategy \
  --category trend \
  --input "Buy when 20 EMA crosses above 50 EMA, sell on cross-under"
```

**Option 2: Manual**
1. Create `src/strategies/custom/my_strategy.py`
2. Extend `BaseStrategy`, use `@register` decorator
3. Restart container: `docker compose restart backtester-v2`

See [docs/strategies/ADDING_STRATEGIES.md](docs/strategies/ADDING_STRATEGIES.md) for the full guide.

## Strategy Interface

```python
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register

@register
class MyStrategy(BaseStrategy):
    name = "my_strategy"
    version = "1.0.0"
    description = "Description here"
    category = "trend"  # trend | mean_reversion | momentum | multi_factor | volatility
    tags = ["trend_following"]

    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow

    def get_warmup_periods(self) -> int:
        return self.slow + 5

    def get_parameter_schema(self) -> dict:
        return {
            "fast": {"type": "integer", "default": 20, "minimum": 5, "maximum": 100},
            "slow": {"type": "integer", "default": 50, "minimum": 10, "maximum": 300},
        }

    def generate_signal(self, window) -> Signal | None:
        hist = window.historical()
        cur = window.current_bar()
        # Your logic here — use polars only, no look-ahead
        return Signal("buy", 1.0, 0.8, {})
```

## Project Structure

```
ava-backtest-api/
├── docker-compose.yml          # Full stack definition
├── Dockerfile                  # Python 3.11 API image
├── pyproject.toml              # Dependencies
├── src/
│   ├── main.py                 # Uvicorn entry point
│   ├── api/v2/
│   │   ├── app.py              # FastAPI app + CORS
│   │   ├── backtests.py        # Backtest CRUD + engine task
│   │   ├── strategies.py       # Strategy listing + detail
│   │   ├── websocket.py        # Live progress WebSocket
│   │   └── models.py           # Pydantic request models
│   ├── core/
│   │   ├── config.py           # Settings (from .env)
│   │   ├── analytics/metrics.py # 20+ performance metrics
│   │   ├── data/
│   │   │   ├── providers/      # FMP, Alpaca, NSEpy
│   │   │   └── universe/       # Symbol lists per market
│   │   ├── execution/
│   │   │   ├── engine.py       # BacktestEngine
│   │   │   └── run_store.py    # In-memory run store
│   │   ├── markets/registry.py # Market definitions (US, IN)
│   │   └── strategy/
│   │       ├── base.py         # BaseStrategy + Signal
│   │       └── registry.py     # @register + get_strategy()
│   └── strategies/
│       ├── classic/            # 9 built-in strategies
│       └── custom/             # 115 auto-discovered strategies
├── docs/
│   ├── strategies/             # Strategy documentation
│   ├── api/                    # API reference
│   └── architecture.md         # System design
└── migrations/                 # Alembic DB migrations
```

## Configuration

All via environment variables in `.env`:

```bash
# Data Providers
FMP_API_KEY=...              # Financial Modeling Prep (required for US market)
ALPACA_API_KEY=...           # Alpaca (alternative US data)
ALPHAVANTAGE_API_KEY=...     # AlphaVantage (backup)
FINNHUB_API_KEY=...          # Finnhub (news/sentiment)

# Database
DATABASE_URL=postgresql+asyncpg://ava:ava2026@ava-db:5432/ava

# Redis
REDIS_URL=redis://redis:6379/0

# Strategy Builder (optional)
OPENROUTER_API_KEY=...       # For AI strategy generation
```

## Development

```bash
# Rebuild after code changes
docker compose build backtester-v2 && docker compose up -d backtester-v2

# View logs
docker compose logs backtester-v2 --tail=50 -f

# Run tests
docker compose exec backtester-v2 pytest

# Add strategy without rebuild (src is volume-mounted)
# Just drop a .py file in src/strategies/custom/ and restart:
docker compose restart backtester-v2
```

## Related Projects

| Project | Description |
|---------|-------------|
| [ava_backtest_ui](https://github.com/noufal85/ava_backtest_ui) | React frontend with TradingView charts |
| [ava_dashboard](https://github.com/noufal85/ava_dashboard) | Personal AI dashboard (Kanban, Journal, Cron, Blog) |
| [trading-backtester](https://github.com/noufal85/trading-backtester) | Original backtester (strategies migrated from here) |

## License

Private — © 2026 Noufal
