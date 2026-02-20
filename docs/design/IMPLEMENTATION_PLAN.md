# V2 Implementation Plan

## Executive Summary

This document outlines the phased implementation plan for migrating Trading Backtester from V1 to V2. The plan breaks the work into 5 phases over ~16-20 weeks, each independently deliverable. It covers migration strategy, testing approach, risk assessment, and success criteria to ensure a smooth transition from V1's monolithic architecture to V2's composable, high-performance platform.

## Phased Build Plan

### Phase 1: Data Layer + Core Abstractions (Weeks 1-4)

**Goal**: Build the foundational data infrastructure and core type system that everything else depends on.

**Deliverables**:
1. Core type system (Signal, MarketData, Order, Fill, Position)
2. Data provider abstraction with FMP adapter
3. Multi-tier caching (Memory → Redis → TimescaleDB)
4. Data validation pipeline
5. Universe management
6. TimescaleDB V2 schema migration

**Implementation Order**:

```python
# Step 1: Core types and protocols (Week 1)
# v2/core/types.py

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Literal
from decimal import Decimal
import polars as pl

@dataclass(frozen=True)
class Signal:
    """Immutable trading signal."""
    timestamp: datetime
    symbol: str
    action: Literal["buy", "sell", "hold", "exit"]
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        assert 0.0 <= self.strength <= 1.0, f"strength must be [0,1], got {self.strength}"
        assert 0.0 <= self.confidence <= 1.0, f"confidence must be [0,1], got {self.confidence}"

@dataclass
class MarketData:
    """Multi-timeframe market data with temporal enforcement."""
    primary_df: pl.DataFrame
    secondary_dfs: dict[str, pl.DataFrame]
    current_idx: int
    symbol: str

    def current_bar(self, timeframe: str | None = None) -> pl.DataFrame:
        df = self.primary_df if timeframe is None else self.secondary_dfs[timeframe]
        if self.current_idx >= len(df):
            raise IndexError(f"current_idx {self.current_idx} out of bounds for {len(df)} bars")
        return df.slice(self.current_idx, 1)

    def historical(self, lookback: int | None = None, timeframe: str | None = None) -> pl.DataFrame:
        df = self.primary_df if timeframe is None else self.secondary_dfs[timeframe]
        end_idx = self.current_idx
        start_idx = max(0, end_idx - (lookback or end_idx))
        return df.slice(start_idx, end_idx - start_idx)

    def indicators_window(self, timeframe: str | None = None) -> pl.DataFrame:
        """Current + historical bars for indicator computation (no future data)."""
        df = self.primary_df if timeframe is None else self.secondary_dfs[timeframe]
        return df.slice(0, self.current_idx + 1)

@dataclass
class Order:
    id: str
    symbol: str
    side: Literal["buy", "sell", "short", "cover"]
    order_type: Literal["market", "limit", "stop", "stop_limit"]
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    strategy_id: str = ""
    metadata: dict = field(default_factory=dict)

@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    fill_quality: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def gross_amount(self) -> float:
        return self.quantity * self.price

    @property
    def net_amount(self) -> float:
        return self.gross_amount + self.commission + abs(self.slippage)
```

```python
# Step 2: Data provider abstraction (Week 1-2)
# v2/core/data/providers/base.py

from abc import ABC, abstractmethod
from datetime import date
import polars as pl

class DataProvider(ABC):
    """Pluggable data source interface."""

    @abstractmethod
    async def fetch_prices(self, symbol: str, timeframe: str,
                          start: date, end: date) -> pl.DataFrame:
        ...

    @abstractmethod
    async def fetch_universe(self, name: str) -> list[str]:
        ...

    async def health_check(self) -> bool:
        """Test provider connectivity."""
        try:
            df = await self.fetch_prices("SPY", "1d",
                                        date.today() - timedelta(days=5),
                                        date.today())
            return not df.is_empty()
        except Exception:
            return False


# v2/core/data/providers/fmp.py
import aiohttp
from aiolimiter import AsyncLimiter

class FMPProvider(DataProvider):
    """Financial Modeling Prep data provider."""

    def __init__(self, api_key: str, rate_limit: int = 300):
        self.api_key = api_key
        self.session: aiohttp.ClientSession | None = None
        self.limiter = AsyncLimiter(rate_limit, 60)

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def fetch_prices(self, symbol: str, timeframe: str,
                          start: date, end: date) -> pl.DataFrame:
        await self._ensure_session()
        async with self.limiter:
            if timeframe == "1d":
                return await self._fetch_daily(symbol, start, end)
            return await self._fetch_intraday(symbol, timeframe, start, end)

    async def _fetch_daily(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        params = {"from": start.isoformat(), "to": end.isoformat(), "apikey": self.api_key}

        async with self.session.get(url, params=params) as resp:
            data = await resp.json()

        if "historical" not in data:
            return pl.DataFrame()

        records = [
            {
                "timestamp": datetime.strptime(item["date"], "%Y-%m-%d"),
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
                "volume": int(item["volume"]),
                "adj_close": float(item.get("adjClose", item["close"])),
            }
            for item in data["historical"]
        ]

        return pl.DataFrame(records).sort("timestamp")

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
```

```python
# Step 3: Multi-tier caching (Week 2)
# v2/core/data/cache.py

import hashlib
from io import BytesIO
from collections import OrderedDict
import polars as pl
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import date

class DataCache:
    """Memory → Redis → TimescaleDB tiered cache."""

    def __init__(self, redis_url: str = "redis://localhost:6379",
                 db_session_factory=None, max_memory_mb: int = 512):
        self.redis = redis.from_url(redis_url)
        self.db_factory = db_session_factory
        self.max_memory_mb = max_memory_mb
        self._mem: OrderedDict[str, pl.DataFrame] = OrderedDict()
        self._mem_bytes = 0

    def _cache_key(self, symbol: str, timeframe: str, start: date, end: date) -> str:
        return f"prices:{symbol}:{timeframe}:{start.isoformat()}:{end.isoformat()}"

    async def get(self, symbol: str, timeframe: str,
                  start: date, end: date) -> pl.DataFrame | None:
        key = self._cache_key(symbol, timeframe, start, end)

        # L1: memory
        if key in self._mem:
            self._mem.move_to_end(key)
            return self._mem[key]

        # L2: Redis
        raw = await self.redis.get(key)
        if raw:
            df = pl.read_parquet(BytesIO(raw))
            self._put_mem(key, df)
            return df

        # L3: TimescaleDB
        if self.db_factory:
            df = await self._load_from_db(symbol, timeframe, start, end)
            if df is not None and not df.is_empty():
                await self._put_redis(key, df)
                self._put_mem(key, df)
                return df

        return None

    async def put(self, symbol: str, timeframe: str,
                  start: date, end: date, df: pl.DataFrame) -> None:
        key = self._cache_key(symbol, timeframe, start, end)
        self._put_mem(key, df)
        await self._put_redis(key, df, ttl=3600)

    def _put_mem(self, key: str, df: pl.DataFrame) -> None:
        size = df.estimated_size()
        while self._mem_bytes + size > self.max_memory_mb * 1024 * 1024 and self._mem:
            _, evicted = self._mem.popitem(last=False)
            self._mem_bytes -= evicted.estimated_size()
        self._mem[key] = df
        self._mem_bytes += size

    async def _put_redis(self, key: str, df: pl.DataFrame, ttl: int = 3600) -> None:
        buf = BytesIO()
        df.write_parquet(buf)
        await self.redis.setex(key, ttl, buf.getvalue())

    async def _load_from_db(self, symbol: str, timeframe: str,
                           start: date, end: date) -> pl.DataFrame | None:
        async with self.db_factory() as session:
            result = await session.execute(
                text("""
                    SELECT timestamp, open, high, low, close, volume, adj_close
                    FROM backtester.prices_v2
                    WHERE symbol = :symbol AND timeframe = :tf
                      AND timestamp >= :start AND timestamp <= :end
                    ORDER BY timestamp
                """),
                {"symbol": symbol, "tf": timeframe,
                 "start": start, "end": end}
            )
            rows = result.fetchall()
            if not rows:
                return None
            return pl.DataFrame(
                [dict(r._mapping) for r in rows]
            )
```

```python
# Step 4: Data validation (Week 2-3)
# v2/core/data/validator.py

from dataclasses import dataclass, field
import polars as pl

@dataclass
class QualityReport:
    symbol: str
    timeframe: str
    record_count: int
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score: float = 1.0

    def is_usable(self, min_score: float = 0.7) -> bool:
        return self.score >= min_score and self.record_count > 0

class DataValidator:
    """Validates OHLCV data quality before strategy consumption."""

    def validate(self, df: pl.DataFrame, symbol: str, timeframe: str) -> QualityReport:
        report = QualityReport(symbol=symbol, timeframe=timeframe, record_count=len(df))

        if df.is_empty():
            report.issues.append("No data")
            report.score = 0.0
            return report

        self._check_ohlc_consistency(df, report)
        self._check_gaps(df, timeframe, report)
        self._check_volume(df, report)
        self._check_outliers(df, report)

        # Score: start at 1.0, subtract penalties
        penalty = len(report.issues) * 0.15 + len(report.warnings) * 0.05
        report.score = max(0.0, min(1.0, 1.0 - penalty))
        return report

    def _check_ohlc_consistency(self, df: pl.DataFrame, report: QualityReport) -> None:
        bad_high = df.filter(pl.col("high") < pl.max_horizontal("open", "close"))
        if len(bad_high) > 0:
            report.issues.append(f"{len(bad_high)} bars where high < max(open, close)")

        bad_low = df.filter(pl.col("low") > pl.min_horizontal("open", "close"))
        if len(bad_low) > 0:
            report.issues.append(f"{len(bad_low)} bars where low > min(open, close)")

        zero_prices = df.filter(
            (pl.col("open") <= 0) | (pl.col("high") <= 0) |
            (pl.col("low") <= 0) | (pl.col("close") <= 0)
        )
        if len(zero_prices) > 0:
            report.issues.append(f"{len(zero_prices)} bars with zero/negative prices")

    def _check_gaps(self, df: pl.DataFrame, timeframe: str, report: QualityReport) -> None:
        if len(df) < 2:
            return
        ts = df["timestamp"].sort()
        diffs = ts.diff().drop_nulls()
        max_gap_map = {"1d": 4, "1h": 6, "15m": 2, "5m": 1}  # in units of expected freq
        # Simplified: just count large gaps
        median_diff = diffs.median()
        if median_diff is not None:
            large_gaps = diffs.filter(diffs > median_diff * max_gap_map.get(timeframe, 3))
            if len(large_gaps) > len(df) * 0.01:
                report.warnings.append(f"{len(large_gaps)} time gaps detected")

    def _check_volume(self, df: pl.DataFrame, report: QualityReport) -> None:
        if "volume" not in df.columns:
            return
        zero_vol_pct = (df["volume"] == 0).mean()
        if zero_vol_pct > 0.05:
            report.warnings.append(f"{zero_vol_pct:.1%} bars with zero volume")

    def _check_outliers(self, df: pl.DataFrame, report: QualityReport) -> None:
        if len(df) < 10:
            return
        returns = df["close"].pct_change().drop_nulls()
        extreme = returns.filter(returns.abs() > 0.5)
        if len(extreme) > 0:
            report.warnings.append(f"{len(extreme)} extreme price moves (>50%)")
```

```sql
-- Step 5: Database migration (Week 3)
-- v2/migrations/001_v2_schema.sql

-- Create V2 schema within backtester namespace
CREATE TABLE IF NOT EXISTS backtester.prices_v2 (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timeframe TEXT NOT NULL,
    open DECIMAL(12,4) NOT NULL,
    high DECIMAL(12,4) NOT NULL,
    low DECIMAL(12,4) NOT NULL,
    close DECIMAL(12,4) NOT NULL,
    volume BIGINT NOT NULL,
    adj_close DECIMAL(12,4),
    data_source TEXT NOT NULL DEFAULT 'fmp',
    quality_score DECIMAL(4,3) DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe, timestamp)
);

SELECT create_hypertable('backtester.prices_v2', 'timestamp',
                         if_not_exists => TRUE);

ALTER TABLE backtester.prices_v2 SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('backtester.prices_v2', INTERVAL '7 days');

CREATE TABLE IF NOT EXISTS backtester.strategy_runs_v2 (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    strategy_hash TEXT NOT NULL,
    param_hash TEXT NOT NULL,
    param_yaml TEXT NOT NULL,
    run_type TEXT NOT NULL DEFAULT 'backtest',
    parent_run_id UUID REFERENCES backtester.strategy_runs_v2(id),
    universe_name TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(15,2) NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_seconds DECIMAL(10,3),
    trades_count INTEGER,
    error_message TEXT,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS backtester.trades_v2 (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES backtester.strategy_runs_v2(id),
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    entry_date DATE NOT NULL,
    entry_price DECIMAL(12,4) NOT NULL,
    exit_date DATE,
    exit_price DECIMAL(12,4),
    shares INTEGER NOT NULL,
    pnl DECIMAL(15,2),
    pnl_pct DECIMAL(8,4),
    commission_cost DECIMAL(10,2),
    slippage_cost DECIMAL(10,2),
    entry_signal JSONB,
    exit_reason TEXT,
    hold_days INTEGER,
    max_favorable_excursion DECIMAL(15,2),
    max_adverse_excursion DECIMAL(15,2),
    regime_at_entry TEXT,
    volatility_at_entry DECIMAL(8,4),
    position_size_pct DECIMAL(6,3),
    fill_quality_score DECIMAL(4,3)
);

SELECT create_hypertable('backtester.trades_v2', 'entry_date',
                         if_not_exists => TRUE);
```

```python
# Step 6: Universe management (Week 3-4)
# v2/core/data/universe.py

from pydantic import BaseModel
from typing import Optional

class UniverseDefinition(BaseModel):
    type: str  # "static", "filter", "index"
    description: str
    symbols: Optional[list[str]] = None
    filters: Optional[dict] = None
    index_name: Optional[str] = None

class UniverseManager:
    BUILTIN = {
        "sp500_liquid": UniverseDefinition(
            type="filter",
            description="S&P 500 constituents with >$10M avg daily volume",
            filters={"min_market_cap": 10_000_000_000, "min_avg_volume": 500_000}
        ),
        "mega_cap": UniverseDefinition(
            type="filter",
            description="Mega-cap stocks (>$200B market cap)",
            filters={"min_market_cap": 200_000_000_000}
        ),
        "test_universe": UniverseDefinition(
            type="static",
            description="Small test universe",
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        ),
    }

    def __init__(self, db_session_factory):
        self.db_factory = db_session_factory

    async def get_symbols(self, universe_name: str) -> list[str]:
        if universe_name in self.BUILTIN:
            defn = self.BUILTIN[universe_name]
            if defn.type == "static":
                return defn.symbols
            return await self._apply_filters(defn.filters)
        # Try loading from DB
        return await self._load_custom_universe(universe_name)

    async def _apply_filters(self, filters: dict) -> list[str]:
        async with self.db_factory() as session:
            conditions = ["is_active = TRUE"]
            params = {}
            if "min_market_cap" in filters:
                conditions.append("market_cap >= :min_mc")
                params["min_mc"] = filters["min_market_cap"]
            if "min_avg_volume" in filters:
                conditions.append("avg_daily_volume >= :min_vol")
                params["min_vol"] = filters["min_avg_volume"]

            where = " AND ".join(conditions)
            result = await session.execute(
                text(f"SELECT symbol FROM backtester.symbols_v2 WHERE {where} ORDER BY symbol"),
                params
            )
            return [row[0] for row in result.fetchall()]
```

**Testing for Phase 1**:
```python
# tests/unit/test_data_cache.py
import pytest
import polars as pl
from v2.core.data.cache import DataCache
from datetime import date

@pytest.fixture
def sample_prices():
    return pl.DataFrame({
        "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 1, 31), eager=True),
        "open": [100.0] * 31,
        "high": [105.0] * 31,
        "low": [95.0] * 31,
        "close": [102.0] * 31,
        "volume": [1_000_000] * 31,
    })

class TestDataCache:
    async def test_memory_cache_hit(self, sample_prices):
        cache = DataCache(redis_url="redis://localhost:6379", max_memory_mb=64)
        await cache.put("AAPL", "1d", date(2024, 1, 1), date(2024, 1, 31), sample_prices)
        result = await cache.get("AAPL", "1d", date(2024, 1, 1), date(2024, 1, 31))
        assert result is not None
        assert len(result) == 31

    async def test_memory_eviction(self):
        cache = DataCache(redis_url="redis://localhost:6379", max_memory_mb=1)
        # Fill cache beyond capacity → oldest entries evicted
        for i in range(100):
            df = pl.DataFrame({"close": list(range(10000))})
            await cache.put(f"SYM{i}", "1d", date(2024, 1, 1), date(2024, 1, 1), df)
        assert cache._mem_bytes <= 1 * 1024 * 1024 * 1.1  # Allow 10% overhead


# tests/unit/test_validator.py
class TestDataValidator:
    def test_empty_data_fails(self):
        validator = DataValidator()
        report = validator.validate(pl.DataFrame(), "AAPL", "1d")
        assert report.score == 0.0
        assert not report.is_usable()

    def test_good_data_passes(self, sample_prices):
        validator = DataValidator()
        report = validator.validate(sample_prices, "AAPL", "1d")
        assert report.score >= 0.8
        assert report.is_usable()

    def test_bad_ohlc_detected(self):
        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "open": [100.0], "high": [90.0],  # high < open → bad
            "low": [95.0], "close": [98.0], "volume": [1000]
        })
        validator = DataValidator()
        report = validator.validate(df, "BAD", "1d")
        assert len(report.issues) > 0
```

**Phase 1 Success Criteria**:
- [ ] DataProvider interface + FMP implementation passing integration tests
- [ ] 3-tier cache with <5ms L1, <20ms L2, <100ms L3 retrieval
- [ ] Data validation catching >95% of known data quality issues
- [ ] V2 schema deployed to TimescaleDB alongside V1
- [ ] Universe manager returning correct symbol lists
- [ ] ≥90% unit test coverage for data layer

---

### Phase 2: Strategy Framework + Indicator Library (Weeks 5-8)

**Goal**: Build the composable strategy framework, indicator library, and position sizing components.

**Deliverables**:
1. BaseStrategy and protocol interfaces
2. Signal generation framework
3. Vectorized indicator library (20+ indicators)
4. Position sizing components (Fixed, Kelly, Volatility-scaled)
5. Risk management middleware
6. Strategy registry with versioning

**Implementation Order**:

```python
# Week 5: Strategy base + indicator library
# v2/core/strategy/base.py

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

@runtime_checkable
class IndicatorProvider(Protocol):
    name: str
    config_hash: str

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        ...

@runtime_checkable
class SignalGenerator(Protocol):
    def generate(self, data: MarketData, indicators: pl.DataFrame) -> Signal:
        ...

class BaseStrategy(ABC):
    name: str
    version: str
    description: str = ""

    @abstractmethod
    def get_indicators(self) -> list[IndicatorProvider]:
        ...

    @abstractmethod
    def get_signal_generator(self) -> SignalGenerator:
        ...

    def get_warmup_periods(self) -> int:
        return 50

    def get_required_timeframes(self) -> list[str]:
        return []

    def generate_signal(self, data: MarketData) -> Signal:
        """Convenience: compute indicators + generate signal for current bar."""
        indicator_dfs = []
        for ind in self.get_indicators():
            result = ind.compute(data.indicators_window())
            indicator_dfs.append(result)

        if indicator_dfs:
            indicators = pl.concat(indicator_dfs, how="horizontal")
        else:
            indicators = pl.DataFrame()

        return self.get_signal_generator().generate(data, indicators)


# v2/core/indicators/trend.py
import polars as pl
import numpy as np
from dataclasses import dataclass

@dataclass
class SMA:
    period: int
    column: str = "close"
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"sma_{self.period}"
        self.config_hash = f"sma:{self.period}:{self.column}"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.select(
            pl.col(self.column).rolling_mean(window_size=self.period).alias(self.name)
        )

@dataclass
class EMA:
    period: int
    column: str = "close"
    name: str = ""
    timeframe: str | None = None

    def __post_init__(self):
        if not self.name:
            self.name = f"ema_{self.period}"
        self.config_hash = f"ema:{self.period}:{self.column}"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.select(
            pl.col(self.column).ewm_mean(span=self.period).alias(self.name)
        )

@dataclass
class MACD:
    fast: int = 12
    slow: int = 26
    signal: int = 9
    column: str = "close"
    name: str = "macd"

    def __post_init__(self):
        self.config_hash = f"macd:{self.fast}:{self.slow}:{self.signal}:{self.column}"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        fast_ema = df[self.column].ewm_mean(span=self.fast)
        slow_ema = df[self.column].ewm_mean(span=self.slow)
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm_mean(span=self.signal)
        histogram = macd_line - signal_line

        return pl.DataFrame({
            f"{self.name}_line": macd_line,
            f"{self.name}_signal": signal_line,
            f"{self.name}_hist": histogram,
        })


# v2/core/indicators/momentum.py
@dataclass
class RSI:
    period: int = 14
    column: str = "close"
    name: str = ""
    timeframe: str | None = None

    def __post_init__(self):
        if not self.name:
            self.name = f"rsi_{self.period}"
        self.config_hash = f"rsi:{self.period}:{self.column}"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        delta = df[self.column].diff()
        gains = delta.clip(lower_bound=0)
        losses = (-delta).clip(lower_bound=0)

        avg_gain = gains.ewm_mean(span=self.period)
        avg_loss = losses.ewm_mean(span=self.period)

        rs = avg_gain / avg_loss.clip(lower_bound=1e-10)
        rsi = 100 - (100 / (1 + rs))

        return pl.DataFrame({self.name: rsi})


# v2/core/indicators/volatility.py
@dataclass
class BollingerBands:
    period: int = 20
    std_dev: float = 2.0
    column: str = "close"
    name: str = "bb"
    timeframe: str | None = None

    def __post_init__(self):
        self.config_hash = f"bb:{self.period}:{self.std_dev}:{self.column}"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        sma = df[self.column].rolling_mean(window_size=self.period)
        std = df[self.column].rolling_std(window_size=self.period)

        return pl.DataFrame({
            f"{self.name}_middle": sma,
            f"{self.name}_upper": sma + self.std_dev * std,
            f"{self.name}_lower": sma - self.std_dev * std,
            f"{self.name}_width": (2 * self.std_dev * std) / sma.clip(lower_bound=1e-10),
        })

@dataclass
class ATR:
    period: int = 14
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"atr_{self.period}"
        self.config_hash = f"atr:{self.period}"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()

        tr = pl.max_horizontal(high_low, high_close, low_close)
        atr = tr.ewm_mean(span=self.period)

        return pl.DataFrame({self.name: atr})
```

```python
# Week 6: Position sizing + risk management
# v2/core/sizing/base.py

from abc import ABC, abstractmethod

class PositionSizer(ABC):
    @abstractmethod
    def calculate_size(self, signal: Signal, portfolio: 'Portfolio',
                      current_price: float) -> int:
        ...

class FixedDollarSizer(PositionSizer):
    def __init__(self, dollars: float = 5000):
        self.dollars = dollars

    def calculate_size(self, signal: Signal, portfolio: 'Portfolio',
                      current_price: float) -> int:
        return max(1, int(self.dollars / current_price))

class KellySizer(PositionSizer):
    def __init__(self, lookback: int = 50, max_kelly: float = 0.25,
                 fallback_pct: float = 0.02):
        self.lookback = lookback
        self.max_kelly = max_kelly
        self.fallback_pct = fallback_pct

    def calculate_size(self, signal: Signal, portfolio: 'Portfolio',
                      current_price: float) -> int:
        recent = portfolio.get_recent_closed_trades(self.lookback)
        if len(recent) < 10:
            return max(1, int(portfolio.equity * self.fallback_pct / current_price))

        wins = [t.pnl for t in recent if t.pnl > 0]
        losses = [abs(t.pnl) for t in recent if t.pnl <= 0]

        if not losses:
            kelly_f = self.fallback_pct
        else:
            win_rate = len(wins) / len(recent)
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses)
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
            kelly_f = win_rate - (1 - win_rate) / win_loss_ratio
            kelly_f = max(0, min(kelly_f, self.max_kelly))

        return max(1, int(portfolio.equity * kelly_f / current_price))

class VolatilityScaledSizer(PositionSizer):
    def __init__(self, target_vol: float = 0.02, lookback: int = 20):
        self.target_vol = target_vol
        self.lookback = lookback

    def calculate_size(self, signal: Signal, portfolio: 'Portfolio',
                      current_price: float) -> int:
        # Requires historical prices in signal metadata
        hist_returns = signal.metadata.get("recent_returns", [])
        if len(hist_returns) < self.lookback:
            return max(1, int(portfolio.equity * 0.02 / current_price))

        current_vol = float(np.std(hist_returns[-self.lookback:]))
        if current_vol <= 0:
            return 0

        vol_scalar = self.target_vol / current_vol
        base_size = portfolio.equity * 0.02 / current_price
        return max(1, int(base_size * vol_scalar))


# v2/core/risk/manager.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class RiskDecision:
    action: str  # "pass", "block", "modify"
    reason: str = ""
    modified_order: Optional[Order] = None

class RiskRule(ABC):
    @abstractmethod
    def evaluate(self, order: Order, portfolio: 'Portfolio') -> RiskDecision:
        ...

class RiskManager:
    def __init__(self):
        self.rules: list[RiskRule] = []

    def add_rule(self, rule: RiskRule) -> 'RiskManager':
        self.rules.append(rule)
        return self

    def evaluate(self, order: Order, portfolio: 'Portfolio') -> RiskDecision:
        for rule in self.rules:
            decision = rule.evaluate(order, portfolio)
            if decision.action == "block":
                return decision
            if decision.action == "modify" and decision.modified_order:
                order = decision.modified_order
        return RiskDecision(action="pass")

class MaxPositionRule(RiskRule):
    def __init__(self, max_pct: float = 0.05):
        self.max_pct = max_pct

    def evaluate(self, order: Order, portfolio: 'Portfolio') -> RiskDecision:
        if order.price is None:
            return RiskDecision(action="pass")
        position_value = order.quantity * order.price
        position_pct = position_value / portfolio.equity if portfolio.equity > 0 else 1.0
        if position_pct > self.max_pct:
            max_shares = max(1, int(portfolio.equity * self.max_pct / order.price))
            modified = Order(**{**order.__dict__, "quantity": max_shares})
            return RiskDecision(action="modify", modified_order=modified)
        return RiskDecision(action="pass")

class MaxExposureRule(RiskRule):
    def __init__(self, max_total_pct: float = 0.80):
        self.max_total_pct = max_total_pct

    def evaluate(self, order: Order, portfolio: 'Portfolio') -> RiskDecision:
        current_exposure = portfolio.gross_exposure / portfolio.equity if portfolio.equity > 0 else 0
        if current_exposure >= self.max_total_pct:
            return RiskDecision(action="block", reason="Max total exposure reached")
        return RiskDecision(action="pass")

class StopLossRule(RiskRule):
    def __init__(self, pct: float = 0.02):
        self.pct = pct

    def evaluate(self, order: Order, portfolio: 'Portfolio') -> RiskDecision:
        if order.price and order.side in ("buy", "cover"):
            stop_price = order.price * (1 - self.pct)
            order.metadata["stop_loss"] = stop_price
        return RiskDecision(action="pass")
```

```python
# Week 7-8: Strategy registry + example strategies
# v2/core/strategy/registry.py

import hashlib
import inspect
from typing import Type

class StrategyRegistry:
    _instance = None
    _strategies: dict[str, dict[str, Type[BaseStrategy]]] = {}

    @classmethod
    def instance(cls) -> 'StrategyRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, strategy_cls: Type[BaseStrategy]) -> None:
        name = strategy_cls.name
        version = strategy_cls.version
        if name not in self._strategies:
            self._strategies[name] = {}
        self._strategies[name][version] = strategy_cls

    def get(self, name: str, version: str = "latest") -> Type[BaseStrategy]:
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' not found")
        versions = self._strategies[name]
        if version == "latest":
            version = max(versions.keys())
        if version not in versions:
            raise KeyError(f"Version '{version}' not found for strategy '{name}'")
        return versions[version]

    def list_all(self) -> dict[str, list[str]]:
        return {name: sorted(versions.keys())
                for name, versions in self._strategies.items()}

def strategy(name: str, version: str):
    """Decorator for automatic strategy registration."""
    def decorator(cls):
        cls.name = name
        cls.version = version
        StrategyRegistry.instance().register(cls)
        return cls
    return decorator


# v2/strategies/classic/ma_crossover.py
from v2.core.strategy.base import BaseStrategy
from v2.core.strategy.registry import strategy
from v2.core.indicators.trend import EMA

@strategy("ema_crossover", "2.0.0")
class EMACrossover(BaseStrategy):
    description = "EMA crossover with confirmation"

    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        self.fast = EMA(period=fast_period, name="ema_fast")
        self.slow = EMA(period=slow_period, name="ema_slow")

    def get_indicators(self):
        return [self.fast, self.slow]

    def get_warmup_periods(self) -> int:
        return self.slow.period + 5

    def get_signal_generator(self):
        return self._generate

    def _generate(self, data: MarketData, indicators: pl.DataFrame) -> Signal:
        if len(indicators) < 2:
            return Signal(timestamp=data.current_bar()["timestamp"].item(),
                         symbol=data.symbol, action="hold",
                         strength=0.0, confidence=1.0, metadata={})

        fast_curr = indicators["ema_fast"][-1]
        fast_prev = indicators["ema_fast"][-2]
        slow_curr = indicators["ema_slow"][-1]
        slow_prev = indicators["ema_slow"][-2]

        ts = data.current_bar()["timestamp"].item()

        # Bullish crossover
        if fast_prev <= slow_prev and fast_curr > slow_curr:
            return Signal(timestamp=ts, symbol=data.symbol, action="buy",
                         strength=min(1.0, (fast_curr - slow_curr) / slow_curr * 100),
                         confidence=0.7, metadata={"trigger": "bullish_crossover"})

        # Bearish crossover
        if fast_prev >= slow_prev and fast_curr < slow_curr:
            return Signal(timestamp=ts, symbol=data.symbol, action="exit",
                         strength=min(1.0, (slow_curr - fast_curr) / slow_curr * 100),
                         confidence=0.7, metadata={"trigger": "bearish_crossover"})

        return Signal(timestamp=ts, symbol=data.symbol, action="hold",
                     strength=0.0, confidence=1.0, metadata={})
```

**Phase 2 Success Criteria**:
- [ ] BaseStrategy interface implemented and tested
- [ ] 20+ indicators with <1ms compute time per 1000 bars
- [ ] 3 position sizers (Fixed, Kelly, Vol-scaled) with unit tests
- [ ] Risk manager blocking invalid trades in test scenarios
- [ ] Strategy registry with versioning working
- [ ] 2+ example strategies (EMA crossover, BB mean reversion) passing tests
- [ ] ≥85% unit test coverage for strategy framework

---

### Phase 3: Execution Engine + Portfolio Management (Weeks 9-12)

**Goal**: Build the backtesting engine with realistic fill simulation, portfolio tracking, and performance analytics.

**Deliverables**:
1. Event-driven execution engine
2. Vectorized execution engine (fast mode)
3. Realistic fill simulation (market impact, slippage)
4. Commission models (IBKR, zero-commission)
5. Portfolio state management with equity curve
6. Performance metrics calculator (Sharpe, drawdown, etc.)

```python
# v2/core/execution/engine.py

class BacktestEngine:
    """Core backtest execution engine supporting both modes."""

    def __init__(self, initial_capital: float = 100_000,
                 commission_model: CommissionModel = None,
                 fill_model: FillModel = None,
                 sizer: PositionSizer = None,
                 risk_manager: RiskManager = None):
        self.initial_capital = initial_capital
        self.commission = commission_model or InteractiveBrokersCommission()
        self.fill_model = fill_model or RealisticFillModel()
        self.sizer = sizer or FixedDollarSizer()
        self.risk_manager = risk_manager or RiskManager()

    async def run(self, strategy: BaseStrategy,
                  symbols: list[str],
                  data: dict[str, dict[str, pl.DataFrame]],
                  progress_callback=None) -> BacktestResult:
        """Run event-driven backtest."""
        portfolio = Portfolio(self.initial_capital)
        all_signals = []
        total_bars = sum(
            len(tfs.get(strategy.get_required_timeframes()[0] if strategy.get_required_timeframes()
                        else "1d", pl.DataFrame()))
            for tfs in data.values()
        )
        bars_processed = 0

        for symbol in symbols:
            symbol_data = data.get(symbol, {})
            if not symbol_data:
                continue

            primary_tf = (strategy.get_required_timeframes() or ["1d"])[0]
            primary_df = symbol_data.get(primary_tf, pl.DataFrame())
            if primary_df.is_empty():
                continue

            secondary = {k: v for k, v in symbol_data.items() if k != primary_tf}
            warmup = strategy.get_warmup_periods()

            for bar_idx in range(warmup, len(primary_df)):
                market_data = MarketData(
                    primary_df=primary_df,
                    secondary_dfs=secondary,
                    current_idx=bar_idx,
                    symbol=symbol
                )

                # Generate signal
                signal = strategy.generate_signal(market_data)
                all_signals.append(signal)

                if signal.action in ("buy", "sell", "exit"):
                    current_price = primary_df["close"][bar_idx]
                    await self._process_signal(signal, portfolio, market_data, current_price)

                # Update portfolio mark-to-market
                portfolio.update_market_values(
                    {symbol: primary_df["close"][bar_idx]}
                )

                bars_processed += 1
                if progress_callback and bars_processed % 100 == 0:
                    pct = int(bars_processed / total_bars * 100)
                    progress_callback(pct, f"Processing {symbol} bar {bar_idx}")

        metrics = PerformanceAnalyzer().calculate(portfolio)
        return BacktestResult(
            portfolio=portfolio,
            metrics=metrics,
            signals=all_signals,
            trades=portfolio.closed_trades
        )

    async def _process_signal(self, signal: Signal, portfolio: Portfolio,
                              market_data: MarketData, current_price: float):
        if signal.action == "exit":
            # Close existing position
            pos = portfolio.get_position(signal.symbol)
            if pos and not pos.is_flat:
                order = Order(
                    id=f"{signal.symbol}_{signal.timestamp}_exit",
                    symbol=signal.symbol,
                    side="sell" if pos.is_long else "cover",
                    order_type="market",
                    quantity=abs(pos.quantity),
                )
                fill = self.fill_model.simulate(order, market_data)
                fill.commission = self.commission.calculate(fill.quantity, fill.price, fill.side)
                portfolio.apply_fill(fill)
            return

        if signal.action == "buy":
            qty = self.sizer.calculate_size(signal, portfolio, current_price)
            if qty <= 0:
                return

            order = Order(
                id=f"{signal.symbol}_{signal.timestamp}_buy",
                symbol=signal.symbol,
                side="buy",
                order_type="market",
                quantity=qty,
                price=current_price,
            )

            decision = self.risk_manager.evaluate(order, portfolio)
            if decision.action == "block":
                return
            if decision.action == "modify" and decision.modified_order:
                order = decision.modified_order

            fill = self.fill_model.simulate(order, market_data)
            fill.commission = self.commission.calculate(fill.quantity, fill.price, fill.side)
            portfolio.apply_fill(fill)
```

**Phase 3 Success Criteria**:
- [ ] Single-symbol backtest in <500ms (vs V1 ~2-3s)
- [ ] 100-symbol backtest in <2 minutes
- [ ] Fill simulation producing realistic slippage estimates
- [ ] Portfolio tracking with correct equity curve
- [ ] Performance metrics matching reference calculations (pyfolio)
- [ ] Deterministic results: same config → identical output
- [ ] ≥85% test coverage for execution engine

---

### Phase 4: API + UI + Optimization (Weeks 13-16)

**Goal**: Build the FastAPI V2 endpoints, React UI dashboard, and parameter optimization framework.

**Deliverables**:
1. FastAPI V2 router with backtest, strategy, and data endpoints
2. WebSocket real-time progress updates
3. Background backtest worker queue
4. React dashboard with dark theme
5. Walk-forward optimization framework
6. Bayesian parameter optimization (scikit-optimize)

```python
# v2/api/v2_router.py

from fastapi import APIRouter, BackgroundTasks, WebSocket
from pydantic import BaseModel

router = APIRouter(prefix="/api/v2")

@router.post("/backtests")
async def create_backtest(request: BacktestRequest,
                         background_tasks: BackgroundTasks):
    run_id = uuid.uuid4()
    # Store run record, queue execution
    background_tasks.add_task(execute_backtest, run_id, request)
    return {"run_id": run_id, "status": "queued",
            "ws_url": f"/api/v2/ws/backtest/{run_id}"}

@router.get("/backtests/{run_id}")
async def get_results(run_id: uuid.UUID):
    # Load from DB
    ...

@router.get("/strategies")
async def list_strategies():
    registry = StrategyRegistry.instance()
    return {"strategies": registry.list_all()}

@router.websocket("/ws/backtest/{run_id}")
async def backtest_ws(websocket: WebSocket, run_id: uuid.UUID):
    await websocket.accept()
    # Stream progress updates
    ...
```

**Phase 4 Success Criteria**:
- [ ] All V1 API endpoints have V2 equivalents
- [ ] WebSocket progress updates working end-to-end
- [ ] React dashboard rendering backtest results
- [ ] Walk-forward analysis completing in <30 minutes for 100 symbols
- [ ] Bayesian optimization finding better params than grid search
- [ ] API response times <200ms for read endpoints

---

### Phase 5: Migration + Validation + Cutover (Weeks 17-20)

**Goal**: Run V1 and V2 side-by-side, validate results match, migrate strategies, and cut over.

**Deliverables**:
1. V1 compatibility adapter
2. Automated V1→V2 strategy converter
3. Regression test suite (V1 vs V2 comparison)
4. Migration runbook
5. V1 deprecation plan

**Detailed in the Migration Strategy section below.**

---

## Migration Strategy

### Side-by-Side Architecture

```
┌─────────────────────────────────────────────────┐
│                   Nginx Proxy                    │
│          /api/v1/* → V1 (port 8200)             │
│          /api/v2/* → V2 (port 8201)             │
└─────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐     ┌─────────────────┐
│   V1 Backend    │     │   V2 Backend    │
│   (port 8200)   │     │   (port 8201)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌─────────────────┐
         │   TimescaleDB   │
         │  (shared, V1+V2 │
         │   schemas)      │
         └─────────────────┘
```

V1 and V2 share the same TimescaleDB instance but use separate tables. V2 reads V1 price data during migration to avoid re-downloading.

### V1 Compatibility Adapter

```python
# v2/compat/v1_adapter.py

import importlib
import sys
from pathlib import Path

class V1CompatAdapter:
    """Run V1 strategies through V2 infrastructure."""

    def __init__(self, v1_strategies_path: str = "/path/to/v1/strategies"):
        self.v1_path = Path(v1_strategies_path)

    def load_v1_strategy(self, strategy_name: str):
        """Dynamically load a V1 strategy class."""
        # Add V1 path to sys.path temporarily
        sys.path.insert(0, str(self.v1_path.parent))
        try:
            module = importlib.import_module(f"strategies.{strategy_name}")
            # V1 strategies follow a naming convention
            class_name = self._to_class_name(strategy_name)
            return getattr(module, class_name)
        finally:
            sys.path.pop(0)

    def wrap_as_v2(self, v1_strategy_class) -> BaseStrategy:
        """Wrap a V1 strategy to conform to V2 interface."""

        class V1Wrapper(BaseStrategy):
            name = f"v1_{v1_strategy_class.__name__}"
            version = "1.0.0-compat"
            description = f"V1 compatibility wrapper for {v1_strategy_class.__name__}"

            def __init__(self):
                self._v1 = v1_strategy_class()
                self._config = self._v1.get_config() if hasattr(self._v1, 'get_config') else {}

            def get_indicators(self):
                # V1 computes indicators inside generate_signals
                # Return empty — we'll handle it in signal generation
                return []

            def get_signal_generator(self):
                return self

            def get_warmup_periods(self) -> int:
                return self._config.get("warmup_periods", 50)

            def generate(self, data: MarketData, indicators: pl.DataFrame) -> Signal:
                # Convert V2 MarketData → V1 pandas DataFrame
                v1_df = data.indicators_window().to_pandas()

                # Run V1 pipeline
                try:
                    ind_df = self._v1.compute_indicators(v1_df, self._config)
                    sig_df = self._v1.generate_signals(ind_df, self._config)

                    # Extract signal for current bar
                    last_row = sig_df.iloc[-1]
                    action = "hold"
                    if last_row.get("signal") == 1:
                        action = "buy"
                    elif last_row.get("signal") == -1:
                        action = "exit"

                    return Signal(
                        timestamp=data.current_bar()["timestamp"].item(),
                        symbol=data.symbol,
                        action=action,
                        strength=abs(float(last_row.get("signal_strength", 0.5))),
                        confidence=0.6,
                        metadata={"source": "v1_compat"}
                    )
                except Exception as e:
                    return Signal(
                        timestamp=data.current_bar()["timestamp"].item(),
                        symbol=data.symbol, action="hold",
                        strength=0.0, confidence=0.0,
                        metadata={"error": str(e)}
                    )

        return V1Wrapper()

    def _to_class_name(self, snake_case: str) -> str:
        return "".join(word.capitalize() for word in snake_case.split("_"))
```

### Regression Validation Framework

```python
# v2/compat/regression.py

from dataclasses import dataclass
import pandas as pd
import polars as pl

@dataclass
class RegressionResult:
    strategy_name: str
    symbol: str
    v1_trades: int
    v2_trades: int
    v1_total_pnl: float
    v2_total_pnl: float
    pnl_diff_pct: float
    trade_count_match: bool
    signal_alignment: float  # % of bars where signals agree
    passed: bool
    details: str = ""

class RegressionValidator:
    """Compare V1 and V2 results for the same strategy + data."""

    TOLERANCE_PNL_PCT = 5.0     # Allow 5% PnL difference (due to fill model)
    TOLERANCE_TRADE_COUNT = 2    # Allow ±2 trades difference
    MIN_SIGNAL_ALIGNMENT = 0.90  # 90% signal agreement

    async def validate_strategy(self, strategy_name: str,
                               symbols: list[str],
                               start_date: date, end_date: date) -> list[RegressionResult]:
        results = []

        v1_adapter = V1CompatAdapter()
        v1_strategy = v1_adapter.load_v1_strategy(strategy_name)
        v2_strategy = StrategyRegistry.instance().get(strategy_name)

        for symbol in symbols:
            # Run V1
            v1_result = await self._run_v1(v1_strategy, symbol, start_date, end_date)

            # Run V2
            v2_result = await self._run_v2(v2_strategy, symbol, start_date, end_date)

            # Compare
            result = self._compare(strategy_name, symbol, v1_result, v2_result)
            results.append(result)

        return results

    def _compare(self, strategy_name: str, symbol: str,
                 v1_result, v2_result) -> RegressionResult:
        v1_pnl = sum(t.pnl for t in v1_result.trades)
        v2_pnl = sum(t.pnl for t in v2_result.trades)
        pnl_diff = abs(v1_pnl - v2_pnl) / abs(v1_pnl) * 100 if v1_pnl != 0 else 0

        trade_diff = abs(len(v1_result.trades) - len(v2_result.trades))

        # Signal alignment: compare bar-by-bar signals
        signal_alignment = self._compute_signal_alignment(
            v1_result.signals, v2_result.signals
        )

        passed = (
            pnl_diff <= self.TOLERANCE_PNL_PCT and
            trade_diff <= self.TOLERANCE_TRADE_COUNT and
            signal_alignment >= self.MIN_SIGNAL_ALIGNMENT
        )

        return RegressionResult(
            strategy_name=strategy_name,
            symbol=symbol,
            v1_trades=len(v1_result.trades),
            v2_trades=len(v2_result.trades),
            v1_total_pnl=v1_pnl,
            v2_total_pnl=v2_pnl,
            pnl_diff_pct=pnl_diff,
            trade_count_match=trade_diff <= self.TOLERANCE_TRADE_COUNT,
            signal_alignment=signal_alignment,
            passed=passed,
            details=f"PnL diff: {pnl_diff:.2f}%, Trade diff: {trade_diff}, "
                    f"Signal alignment: {signal_alignment:.1%}"
        )
```

### Migration Runbook

```markdown
## Migration Steps (Operator Runbook)

### Pre-Migration (Week 17)
1. Deploy V2 backend on port 8201 alongside V1
2. Run V2 schema migration on TimescaleDB
3. Backfill prices_v2 from V1 price tables
4. Validate data integrity: row counts, date ranges, spot-check prices

### Strategy Migration (Weeks 18-19)
For each of the 26+ V1 strategies:
1. Run V1 strategy on test universe (5 symbols, 2 years)
2. Run V2 equivalent (or V1-wrapped) on same data
3. Compare results using RegressionValidator
4. If regression passes: mark strategy as V2-ready
5. If regression fails: investigate, fix, re-run

Priority order:
- Tier 1 (Week 18): Top 10 strategies by usage
- Tier 2 (Week 19): Remaining strategies

### Validation Gate (Week 19)
- [ ] ≥90% of strategies passing regression
- [ ] V2 performance ≥ V1 (latency, memory)
- [ ] All V1 API endpoints have V2 equivalents
- [ ] UI functional with V2 backend

### Cutover (Week 20)
1. Update nginx to route /api/* → V2 (keep /api/v1/* → V1 as fallback)
2. Update React UI to point to V2 endpoints
3. Monitor for 48 hours
4. If stable: deprecate V1 endpoints (return 301 redirects)
5. If issues: rollback nginx config (instant)

### Post-Cutover
- Keep V1 running for 30 days (read-only, no new backtests)
- Export any V1-only results to V2 tables
- Decommission V1 after 30 days
```

---

## Testing Strategy

### Test Pyramid

```
                 ┌──────────┐
                 │   E2E    │  ~20 tests
                 │  Tests   │  Full backtest → API → UI
                 ├──────────┤
              ┌──┤Integration│  ~100 tests
              │  │  Tests   │  Multi-component, DB, Redis
              │  ├──────────┤
           ┌──┤  │  Unit    │  ~500+ tests
           │  │  │  Tests   │  Isolated components
           │  │  └──────────┘
           └──┘
```

### Unit Tests

```python
# tests/unit/test_indicators.py
import pytest
import polars as pl
import numpy as np
from v2.core.indicators.trend import SMA, EMA, MACD
from v2.core.indicators.momentum import RSI
from v2.core.indicators.volatility import BollingerBands, ATR

class TestSMA:
    def test_basic_sma(self):
        df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
        sma = SMA(period=3)
        result = sma.compute(df)
        assert result["sma_3"][2] == pytest.approx(2.0)
        assert result["sma_3"][4] == pytest.approx(4.0)

    def test_sma_insufficient_data(self):
        df = pl.DataFrame({"close": [1.0, 2.0]})
        sma = SMA(period=5)
        result = sma.compute(df)
        assert result["sma_5"].null_count() == 2  # All null

    def test_sma_performance(self, benchmark):
        df = pl.DataFrame({"close": np.random.randn(100_000).tolist()})
        sma = SMA(period=20)
        benchmark(sma.compute, df)  # Should be <10ms

class TestRSI:
    def test_rsi_bounds(self):
        df = pl.DataFrame({"close": np.random.uniform(90, 110, 200).tolist()})
        rsi = RSI(period=14)
        result = rsi.compute(df)
        valid = result["rsi_14"].drop_nulls()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_rsi_overbought(self):
        # Consistently rising prices → RSI near 100
        prices = [100 + i * 0.5 for i in range(50)]
        df = pl.DataFrame({"close": prices})
        rsi = RSI(period=14)
        result = rsi.compute(df)
        assert result["rsi_14"][-1] > 70

# tests/unit/test_portfolio.py
class TestPortfolio:
    def test_initial_state(self):
        p = Portfolio(100_000)
        assert p.equity == 100_000
        assert p.cash == 100_000
        assert len(p.positions) == 0

    def test_buy_and_sell(self):
        p = Portfolio(100_000)
        buy_fill = Fill("o1", "AAPL", "buy", 100, 150.0,
                       datetime.now(), commission=1.0)
        p.apply_fill(buy_fill)

        assert p.cash == 100_000 - (100 * 150.0) - 1.0
        assert "AAPL" in p.positions
        assert p.positions["AAPL"].quantity == 100

        sell_fill = Fill("o2", "AAPL", "sell", 100, 160.0,
                        datetime.now(), commission=1.0)
        p.apply_fill(sell_fill)

        assert p.cash == pytest.approx(100_000 + (100 * 10.0) - 2.0)
        assert "AAPL" not in p.positions

    def test_no_negative_positions(self):
        p = Portfolio(100_000)
        with pytest.raises(ValueError):
            sell_fill = Fill("o1", "AAPL", "sell", 100, 150.0,
                           datetime.now())
            p.apply_fill(sell_fill)  # Can't sell what you don't own
```

### Integration Tests

```python
# tests/integration/test_full_backtest.py
import pytest
from v2.core.execution.engine import BacktestEngine
from v2.strategies.classic.ma_crossover import EMACrossover

@pytest.fixture
async def real_data():
    """Load actual price data from TimescaleDB."""
    cache = DataCache(redis_url="redis://localhost:6379")
    data = {"AAPL": {"1d": await cache.get("AAPL", "1d",
                                           date(2022, 1, 1), date(2023, 12, 31))}}
    return data

class TestFullBacktest:
    @pytest.mark.integration
    async def test_ema_crossover_runs(self, real_data):
        engine = BacktestEngine(initial_capital=100_000)
        strategy = EMACrossover(fast_period=9, slow_period=21)

        result = await engine.run(strategy, ["AAPL"], real_data)

        assert result.metrics["total_trades"] > 0
        assert result.metrics["total_return"] is not None
        assert len(result.portfolio._equity_history) > 0

    @pytest.mark.integration
    async def test_deterministic_results(self, real_data):
        engine = BacktestEngine(initial_capital=100_000)
        strategy = EMACrossover(fast_period=9, slow_period=21)

        r1 = await engine.run(strategy, ["AAPL"], real_data)
        r2 = await engine.run(strategy, ["AAPL"], real_data)

        assert r1.metrics["total_return"] == r2.metrics["total_return"]
        assert r1.metrics["total_trades"] == r2.metrics["total_trades"]

    @pytest.mark.integration
    async def test_no_lookahead_bias(self, real_data):
        """Verify truncating data doesn't change past signals."""
        engine = BacktestEngine(initial_capital=100_000)
        strategy = EMACrossover()

        # Full data
        full_result = await engine.run(strategy, ["AAPL"], real_data)

        # Truncated data (first 6 months only)
        truncated = {
            "AAPL": {"1d": real_data["AAPL"]["1d"].filter(
                pl.col("timestamp") < datetime(2022, 7, 1)
            )}
        }
        partial_result = await engine.run(strategy, ["AAPL"], truncated)

        # Signals in the overlapping period should be identical
        full_signals = [s for s in full_result.signals
                       if s.timestamp < datetime(2022, 7, 1)]
        assert len(full_signals) == len(partial_result.signals)
        for fs, ps in zip(full_signals, partial_result.signals):
            assert fs.action == ps.action
            assert fs.strength == pytest.approx(ps.strength, abs=1e-6)
```

### Regression Tests (V1 vs V2)

```python
# tests/regression/test_v1_v2_parity.py

V1_STRATEGIES = [
    "ema_crossover", "bb_mean_reversion", "rsi_momentum",
    "macd_divergence", "gap_fade", "opening_range_breakout",
    # ... all 26+ V1 strategies
]

TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
TEST_PERIOD = (date(2022, 1, 1), date(2023, 12, 31))

@pytest.mark.regression
@pytest.mark.parametrize("strategy_name", V1_STRATEGIES)
async def test_v1_v2_parity(strategy_name):
    validator = RegressionValidator()
    results = await validator.validate_strategy(
        strategy_name, TEST_SYMBOLS, *TEST_PERIOD
    )

    for result in results:
        assert result.passed, (
            f"{strategy_name}/{result.symbol}: {result.details}"
        )
```

### Performance Benchmarks

```python
# tests/benchmarks/test_performance.py

@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    async def test_single_symbol_speed(self, benchmark):
        """Single symbol, 1 year daily → target <500ms."""
        engine = BacktestEngine()
        strategy = EMACrossover()
        data = await load_test_data("AAPL", "1d", 1)  # 1 year

        result = benchmark(lambda: asyncio.run(
            engine.run(strategy, ["AAPL"], data)
        ))
        assert benchmark.stats["mean"] < 0.5  # <500ms

    async def test_100_symbols_speed(self, benchmark):
        """100 symbols, 1 year → target <2 minutes."""
        engine = BacktestEngine()
        strategy = EMACrossover()
        symbols = await get_test_universe(100)
        data = await load_test_data_multi(symbols, "1d", 1)

        result = benchmark(lambda: asyncio.run(
            engine.run(strategy, symbols, data)
        ))
        assert benchmark.stats["mean"] < 120  # <2 minutes

    async def test_memory_usage(self):
        """100 symbols should use <2GB."""
        import tracemalloc
        tracemalloc.start()

        engine = BacktestEngine()
        strategy = EMACrossover()
        symbols = await get_test_universe(100)
        data = await load_test_data_multi(symbols, "1d", 4)  # 4 years

        await engine.run(strategy, symbols, data)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak / 1024 / 1024 < 2000  # <2GB peak
```

---

## Timeline Estimates

| Phase | Duration | Weeks | Key Milestone |
|-------|----------|-------|---------------|
| **Phase 1**: Data Layer + Core Abstractions | 4 weeks | 1-4 | V2 data pipeline operational |
| **Phase 2**: Strategy Framework + Indicators | 4 weeks | 5-8 | First V2 strategy backtesting |
| **Phase 3**: Execution Engine + Portfolio | 4 weeks | 9-12 | Full backtest pipeline working |
| **Phase 4**: API + UI + Optimization | 4 weeks | 13-16 | V2 accessible via API/UI |
| **Phase 5**: Migration + Validation + Cutover | 4 weeks | 17-20 | V2 in production, V1 deprecated |
| **Total** | **~20 weeks** | | |

### Parallelization Opportunities

- Phase 2 indicator library can start in Week 3 (partial overlap with Phase 1)
- Phase 4 UI work can start in Week 10 (mocking V2 API)
- Total calendar time could compress to **16-18 weeks** with parallel streams

### Critical Path

```
Phase 1 (core types + data) → Phase 2 (strategy) → Phase 3 (engine) → Phase 5 (migration)
                                                        ↘
                                                     Phase 4 (API/UI) ─→ Phase 5
```

Phase 4 is off critical path — it can slip without delaying migration.

---

## Risk Assessment

### High Risk

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **V1/V2 result divergence** | Strategy trust broken | Medium | Comprehensive regression tests; tolerance-based comparison; investigate every mismatch |
| **Polars migration complexity** | Delays, bugs | Medium | Keep Pandas fallback path; migrate gradually; use `.to_pandas()` bridge |
| **Performance regression** | V2 slower than V1 | Low | Benchmark every PR; profile hot paths; keep vectorized fast-path |

### Medium Risk

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Scope creep** | Timeline overrun | High | Strict phase gating; cut scope not quality; defer optimization features |
| **Redis dependency issues** | Cache layer broken | Low | DataCache works without Redis (degrades to L1+L3); Redis is optional |
| **TimescaleDB schema conflicts** | Data corruption | Low | Separate V2 tables; never modify V1 tables; migration rollback scripts |
| **FMP API changes/outages** | Data pipeline broken | Medium | Multi-provider fallback; local data cache survives outages |

### Low Risk

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Python 3.11+ requirement** | Deployment issues | Very Low | Already using 3.11+; no exotic features |
| **React UI complexity** | Frontend delays | Medium | Phase 4 is off critical path; can ship with API-only initially |
| **scikit-optimize deprecation** | Optimization broken | Low | Can swap to Optuna as alternative; abstracted behind interface |

### Risk Monitoring

```python
# v2/monitoring/risk_dashboard.py
# Lightweight check that runs nightly

RISK_CHECKS = [
    ("v2_test_suite", "pytest tests/ -x --timeout=300", "All tests pass"),
    ("v1_v2_regression", "pytest tests/regression/ --timeout=600", "V1/V2 parity"),
    ("performance_bench", "pytest tests/benchmarks/ --benchmark-only", "No regression"),
    ("data_freshness", "python -m v2.scripts.check_data_freshness", "Data <24h old"),
]
```

---

## Dependencies

### New Python Packages

| Package | Version | Purpose | Phase |
|---------|---------|---------|-------|
| `polars` | ≥0.20 | Fast DataFrame operations | 1 |
| `pydantic` | ≥2.0 | Configuration validation | 1 |
| `redis[asyncio]` | ≥5.0 | L2 cache | 1 |
| `aiohttp` | ≥3.9 | Async HTTP client for providers | 1 |
| `aiolimiter` | ≥1.1 | Rate limiting for API calls | 1 |
| `scikit-optimize` | ≥0.9 | Bayesian parameter optimization | 4 |
| `prometheus-client` | ≥0.20 | Metrics export | 4 |
| `ruff` | ≥0.3 | Linting + formatting | 1 |
| `pytest-xdist` | ≥3.5 | Parallel test execution | 1 |
| `pytest-benchmark` | ≥4.0 | Performance benchmarks | 3 |

### Infrastructure Changes

| Component | Change | Phase |
|-----------|--------|-------|
| **Redis** | New container (or add to existing docker-compose) | 1 |
| **TimescaleDB** | Add V2 tables to existing ava-db (port 5435) | 1 |
| **Nginx** | Reverse proxy for V1+V2 co-hosting | 5 |
| **Docker Compose** | Add V2 backend service on port 8201 | 4 |

### Docker Compose Addition

```yaml
# docker-compose.v2.yml (extends existing)
services:
  backtester-v2:
    build:
      context: .
      dockerfile: Dockerfile.v2
    ports:
      - "8201:8201"
    environment:
      - DATABASE_URL=postgresql://ava:ava2026@ava-db:5432/ava
      - REDIS_URL=redis://redis:6379
      - FMP_API_KEY=${FMP_API_KEY}
    depends_on:
      - ava-db
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### Existing Dependencies (Keep)

- **FastAPI** ≥0.100 (already in V1)
- **SQLAlchemy** ≥2.0 with async (already in V1)
- **pandas** (keep for V1 compat layer, gradually reduce usage)
- **numpy** (keep, used by indicators)
- **TimescaleDB** 2.24 on PostgreSQL 15 (existing ava-db)
- **Docker** (existing infrastructure)

---

## Success Criteria

### Phase 1 — Data Layer
- [ ] DataCache L1 retrieval <5ms, L2 <20ms, L3 <100ms
- [ ] Data validator catches known bad data patterns (zero prices, OHLC violations)
- [ ] FMP provider fetches and caches 100 symbols in <60 seconds
- [ ] V2 schema deployed with zero impact on V1
- [ ] ≥90% unit test coverage

### Phase 2 — Strategy Framework
- [ ] BaseStrategy interface fully defined with type hints
- [ ] 20+ indicators compute in <1ms per 1000 bars
- [ ] Position sizers produce correct sizes across edge cases
- [ ] Risk manager blocks oversized positions, excessive exposure
- [ ] Strategy registry supports version lookup
- [ ] ≥85% test coverage

### Phase 3 — Execution Engine
- [ ] Single-symbol 1-year backtest in <500ms
- [ ] 100-symbol 4-year backtest in <2 minutes
- [ ] Peak memory <2GB for largest backtest
- [ ] Deterministic results (bitwise identical across runs)
- [ ] Fill simulation accounts for spread + market impact
- [ ] Sharpe/drawdown calculations match pyfolio reference
- [ ] ≥85% test coverage

### Phase 4 — API & UI
- [ ] All V1 API endpoints have V2 equivalents
- [ ] WebSocket backtest progress updates at 5% intervals
- [ ] React dashboard renders equity curves, trade tables, metrics
- [ ] Walk-forward analysis completes for 100 symbols in <30 minutes
- [ ] API response times <200ms (p95) for read endpoints
- [ ] Dark theme UI matches design spec

### Phase 5 — Migration & Cutover
- [ ] ≥90% of V1 strategies passing regression validation
- [ ] V1 compatibility adapter runs all 26+ strategies
- [ ] Zero downtime cutover via nginx routing
- [ ] V2 backtest results stored alongside V1 in same DB
- [ ] Rollback possible within 5 minutes
- [ ] 30-day V1 deprecation period before decommission

### Overall V2 Success Metrics
- [ ] **10x faster** single-symbol backtests (200ms vs 2-3s)
- [ ] **5x faster** multi-symbol backtests (2min vs 10min for 100 symbols)
- [ ] **50% less memory** for large backtests
- [ ] **Zero look-ahead bias** verified by temporal constraint tests
- [ ] **<10 lines** to implement a basic strategy
- [ ] **100% backward compatibility** during migration period
- [ ] **Walk-forward + Monte Carlo** analysis available
- [ ] **Real-time WebSocket** updates for running backtests

---

## Appendix: Quick Reference

### Key File Locations

```
v2/
├── core/
│   ├── types.py              # Signal, Order, Fill, Position
│   ├── data/
│   │   ├── providers/base.py # DataProvider ABC
│   │   ├── providers/fmp.py  # FMP implementation
│   │   ├── cache.py          # Multi-tier cache
│   │   ├── validator.py      # Data quality checks
│   │   └── universe.py       # Universe management
│   ├── strategy/
│   │   ├── base.py           # BaseStrategy, protocols
│   │   └── registry.py       # Strategy versioning
│   ├── indicators/
│   │   ├── trend.py          # SMA, EMA, MACD
│   │   ├── momentum.py       # RSI, Stochastic
│   │   └── volatility.py     # BB, ATR
│   ├── sizing/
│   │   ├── fixed.py          # FixedDollarSizer
│   │   ├── kelly.py          # KellySizer
│   │   └── volatility.py     # VolatilityScaledSizer
│   ├── risk/
│   │   └── manager.py        # RiskManager + rules
│   └── execution/
│       ├── engine.py         # BacktestEngine
│       ├── fills.py          # Fill simulation
│       ├── costs.py          # Commission models
│       └── portfolio.py      # Portfolio state
├── compat/
│   ├── v1_adapter.py         # V1 compatibility wrapper
│   └── regression.py         # V1 vs V2 comparison
├── api/
│   └── v2_router.py          # FastAPI V2 endpoints
├── strategies/
│   └── classic/              # Migrated strategies
├── migrations/
│   └── 001_v2_schema.sql     # TimescaleDB V2 tables
└── tests/
    ├── unit/                 # ~500+ tests
    ├── integration/          # ~100 tests
    ├── regression/           # V1/V2 parity
    └── benchmarks/           # Performance tests
```

### Development Commands

```bash
# Run all tests
pytest tests/ -v --timeout=300

# Run only unit tests (fast)
pytest tests/unit/ -v -x

# Run regression suite
pytest tests/regression/ -v --timeout=600

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only --benchmark-json=results.json

# Lint + format
ruff check v2/ --fix
ruff format v2/

# Run V2 backend
uvicorn v2.api.main:app --host 0.0.0.0 --port 8201 --reload

# Database migration
psql -h localhost -p 5435 -U ava -d ava -f v2/migrations/001_v2_schema.sql
```

This implementation plan provides a clear, phased path from V1 to V2 with measurable milestones, comprehensive testing, and safe migration. Each phase is independently deliverable and adds value, so the project can be paused or re-prioritized at any phase boundary without losing progress.
