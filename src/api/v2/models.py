"""Pydantic request/response models — matches docs/api/openapi.yaml."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── Markets ──────────────────────────────────────────────────────────────


class Market(BaseModel):
    code: str
    name: str
    currency: str
    currency_symbol: str
    default_universe: str
    is_default: bool = False


# ── Backtests ────────────────────────────────────────────────────────────


class CreateBacktestRequest(BaseModel):
    strategy_name: str
    strategy_version: str = "latest"
    parameters: dict = Field(default_factory=dict)
    universe: str
    start_date: str
    end_date: str
    initial_capital: float
    market: str = "US"
    sizing: dict | None = None
    risk_rules: list[dict] | None = None
    regime_filter: dict | None = None


class BacktestSummary(BaseModel):
    id: str
    strategy_name: str
    strategy_version: str = "latest"
    universe_name: str = ""
    status: str = "pending"
    total_return_pct: float | None = None
    sharpe_ratio: float | None = None
    created_at: str = ""
    duration_seconds: float | None = None


class Backtest(BaseModel):
    id: str
    strategy_name: str
    strategy_version: str = "latest"
    param_yaml: str = ""
    universe_name: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 0
    status: str = "pending"
    market_code: str = "US"
    created_at: str = ""


class BacktestResults(BaseModel):
    total_return_pct: float = 0
    cagr_pct: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    calmar_ratio: float = 0
    max_drawdown_pct: float = 0
    max_drawdown_days: int = 0
    win_rate_pct: float = 0
    profit_factor: float = 0
    total_trades: int = 0
    avg_hold_days: float = 0
    final_equity: float = 0
    monthly_returns: dict = Field(default_factory=dict)


class BacktestDetail(Backtest):
    results: BacktestResults | None = None


# ── Trades & Equity ─────────────────────────────────────────────────────


class Trade(BaseModel):
    id: str = ""
    symbol: str = ""
    direction: str = "long"
    entry_date: str = ""
    entry_price: float = 0
    exit_date: str | None = None
    exit_price: float | None = None
    shares: int = 0
    pnl: float | None = None
    pnl_pct: float | None = None
    hold_days: int | None = None
    exit_reason: str | None = None


class EquityPoint(BaseModel):
    date: str
    equity: float
    drawdown_pct: float = 0


class EquityCurveResponse(BaseModel):
    points: list[EquityPoint] = Field(default_factory=list)


# ── Strategies ───────────────────────────────────────────────────────────


class StrategySummary(BaseModel):
    name: str
    version: str = ""
    description: str = ""
    category: str = ""
    tags: list[str] = Field(default_factory=list)


class StrategyDetail(BaseModel):
    name: str
    version: str = ""
    description: str = ""
    category: str = ""
    tags: list[str] = Field(default_factory=list)
    default_config: dict = Field(default_factory=dict)
    parameter_schema: dict = Field(default_factory=dict)
    warmup_periods: int = 0
    changelog: str = ""


# ── Universes ────────────────────────────────────────────────────────────


class Universe(BaseModel):
    name: str
    description: str = ""
    type: str = "index"
    symbol_count: int = 0


# ── Data ─────────────────────────────────────────────────────────────────


class Candle(BaseModel):
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


class CandleResponse(BaseModel):
    symbol: str
    market: str = "US"
    timeframe: str = "1d"
    candles: list[Candle] = Field(default_factory=list)


# ── Analytics ────────────────────────────────────────────────────────────


class CorrelationResponse(BaseModel):
    labels: list[str] = Field(default_factory=list)
    matrix: list[list[float]] = Field(default_factory=list)
    market: str = "US"


class PortfolioMetrics(BaseModel):
    combined_return_pct: float = 0
    combined_sharpe: float = 0
    combined_max_drawdown_pct: float = 0
    diversification_ratio: float = 0
    weights: dict = Field(default_factory=dict)
    equity_curve: list[dict] = Field(default_factory=list)
    market: str = "US"


# ── Generic ──────────────────────────────────────────────────────────────


class PaginatedResponse(BaseModel):
    items: list = Field(default_factory=list)
    total: int = 0


class ErrorResponse(BaseModel):
    error: str
    code: str
    details: dict = Field(default_factory=dict)
