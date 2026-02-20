# V2 Strategy Pattern Design

## Overview

The V2 strategy framework transforms from V1's monolithic approach to a composable, modular system where strategies, position sizing, risk management, and execution are separate, pluggable components. This eliminates code duplication and makes strategies easier to test, debug, and optimize.

## Core Philosophy: Separation of Concerns

### V1 Problems
- Strategies handle their own position sizing (code duplication)
- Risk management embedded in each strategy
- Multi-timeframe logic scattered and inconsistent
- Hard to test signals in isolation
- Look-ahead bias prevention relies on developer discipline
- No clear separation between signal generation and trade execution

### V2 Solution: Component Pipeline
```
Data → Indicators → Signals → Sizing → Risk → Execution → Attribution
```

Each stage is a separate, testable component with well-defined interfaces.

## New Base Strategy Architecture

### Core Interfaces

```python
from abc import ABC, abstractmethod
from typing import Protocol, Union
import polars as pl
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Signal:
    """A trading signal at a specific point in time."""
    timestamp: datetime
    symbol: str
    action: str  # "buy", "sell", "hold", "exit"
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    metadata: dict
    
    def __post_init__(self):
        assert 0.0 <= self.strength <= 1.0
        assert 0.0 <= self.confidence <= 1.0
        assert self.action in ["buy", "sell", "hold", "exit"]

@dataclass  
class MarketData:
    """Multi-timeframe market data with temporal constraints."""
    primary_df: pl.DataFrame  # Primary timeframe (e.g., 15m)
    secondary_dfs: dict[str, pl.DataFrame]  # {"1d": daily_df, "1h": hourly_df}
    current_idx: int  # Current bar index
    symbol: str
    
    def current_bar(self, timeframe: str = None) -> pl.DataFrame:
        """Get current bar (no lookahead)."""
        df = self.primary_df if timeframe is None else self.secondary_dfs[timeframe]
        return df.slice(self.current_idx, 1)
    
    def historical(self, lookback: int = None, timeframe: str = None) -> pl.DataFrame:
        """Get historical data up to (but not including) current bar."""
        df = self.primary_df if timeframe is None else self.secondary_dfs[timeframe]
        end_idx = self.current_idx
        start_idx = max(0, end_idx - (lookback or end_idx))
        return df.slice(start_idx, end_idx - start_idx)

class IndicatorProvider(Protocol):
    """Interface for indicator computation."""
    
    def compute(self, data: MarketData) -> pl.DataFrame:
        """Compute indicators for the given market data."""
        ...

class SignalGenerator(Protocol):
    """Interface for signal generation."""
    
    def generate(self, data: MarketData, indicators: pl.DataFrame) -> Signal:
        """Generate a signal for the current bar."""
        ...

class BaseStrategy(ABC):
    """V2 base strategy - composable and testable."""
    
    # Metadata
    name: str
    version: str
    description: str
    
    # Required components
    @abstractmethod
    def get_indicators(self) -> list[IndicatorProvider]:
        """Return list of required indicators."""
        ...
    
    @abstractmethod
    def get_signal_generator(self) -> SignalGenerator:
        """Return the signal generation logic."""
        ...
    
    # Optional components (defaults provided)
    def get_warmup_periods(self) -> int:
        """Bars needed before strategy can generate signals."""
        return 50
    
    def get_required_timeframes(self) -> list[str]:
        """Additional timeframes needed (beyond primary)."""
        return []
    
    def validate_data(self, data: MarketData) -> bool:
        """Pre-flight data quality checks."""
        return not data.primary_df.is_empty()
```

### Example V2 Strategy Implementation

```python
from v2.core.strategy.base import BaseStrategy, Signal, MarketData
from v2.core.indicators import BollingerBands, RSI
from v2.core.strategy.signals import CombinedSignalGenerator

class BollingerMeanReversion(BaseStrategy):
    """V2 Bollinger Bands mean reversion strategy."""
    
    name = "bollinger_mean_reversion"
    version = "2.0.0"
    description = "BB mean reversion with RSI confirmation"
    
    def __init__(self, bb_period=20, bb_std=2.0, rsi_period=14, 
                 rsi_oversold=30, rsi_overbought=70):
        self.bb = BollingerBands(period=bb_period, std_dev=bb_std)
        self.rsi = RSI(period=rsi_period)
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def get_indicators(self) -> list[IndicatorProvider]:
        return [self.bb, self.rsi]
    
    def get_signal_generator(self) -> SignalGenerator:
        return CombinedSignalGenerator([
            self._bb_signal_logic,
            self._rsi_confirmation
        ])
    
    def _bb_signal_logic(self, data: MarketData, indicators: pl.DataFrame) -> Signal:
        """BB-based entry/exit signals."""
        current = data.current_bar()
        current_low = current['low'].item()
        current_high = current['high'].item()
        
        bb_lower = indicators['bb_lower'].tail(1).item()
        bb_upper = indicators['bb_upper'].tail(1).item()
        
        if current_low < bb_lower:
            return Signal(
                timestamp=current['timestamp'].item(),
                symbol=data.symbol,
                action="buy",
                strength=min(1.0, (bb_lower - current_low) / bb_lower),
                confidence=0.8,
                metadata={"trigger": "bb_lower_touch", "bb_lower": bb_lower}
            )
        elif current_high > bb_upper:
            return Signal(
                timestamp=current['timestamp'].item(),
                symbol=data.symbol,
                action="exit",
                strength=min(1.0, (current_high - bb_upper) / bb_upper),
                confidence=0.8,
                metadata={"trigger": "bb_upper_touch", "bb_upper": bb_upper}
            )
        
        return Signal(
            timestamp=current['timestamp'].item(),
            symbol=data.symbol,
            action="hold",
            strength=0.0,
            confidence=1.0,
            metadata={}
        )
```

## Multi-Timeframe Support (Native)

### V1 Limitations
- `candle_interval` vs `indicator_interval` confusion
- Manual alignment of timeframes
- Indicators computed on wrong timeframe by accident

### V2 Native Multi-Timeframe

```python
class MultiTimeframeStrategy(BaseStrategy):
    """Base for strategies using multiple timeframes."""
    
    def get_required_timeframes(self) -> list[str]:
        """Define all needed timeframes."""
        return ["1d", "4h", "15m"]  # Daily trend, 4h momentum, 15m entries
    
    def get_indicators(self) -> list[IndicatorProvider]:
        return [
            # Daily indicators
            EMA(period=50, timeframe="1d", name="trend_ema"),
            
            # 4-hour indicators  
            RSI(period=14, timeframe="4h", name="momentum_rsi"),
            
            # 15-minute indicators
            BollingerBands(period=20, timeframe="15m", name="entry_bb"),
        ]

# Engine automatically handles alignment
class MultiTimeframeEngine:
    def process_symbol(self, symbol: str, strategy: MultiTimeframeStrategy):
        # Load all required timeframes
        data = {}
        for tf in strategy.get_required_timeframes():
            data[tf] = self.data_provider.load(symbol, tf, start_date, end_date)
        
        # Align to primary timeframe
        market_data = self._align_timeframes(data, primary_tf="15m")
        
        # Execute strategy bar by bar
        for bar_idx in range(strategy.get_warmup_periods(), len(market_data.primary_df)):
            market_data.current_idx = bar_idx
            signal = strategy.generate_signal(market_data)
            # ... continue pipeline
```

## Position Sizing Framework

### Pluggable Sizers

```python
from abc import ABC, abstractmethod

class PositionSizer(ABC):
    """Base class for position sizing algorithms."""
    
    @abstractmethod
    def calculate_size(self, signal: Signal, portfolio: Portfolio, 
                      market_data: MarketData) -> int:
        """Calculate number of shares for this signal."""
        ...

class FixedDollarSizer(PositionSizer):
    def __init__(self, target_dollars: float):
        self.target_dollars = target_dollars
    
    def calculate_size(self, signal: Signal, portfolio: Portfolio, 
                      market_data: MarketData) -> int:
        current_price = market_data.current_bar()['close'].item()
        return int(self.target_dollars / current_price)

class KellySizer(PositionSizer):
    def __init__(self, lookback_trades: int = 50, max_kelly_pct: float = 0.25):
        self.lookback_trades = lookback_trades
        self.max_kelly_pct = max_kelly_pct
    
    def calculate_size(self, signal: Signal, portfolio: Portfolio, 
                      market_data: MarketData) -> int:
        # Calculate Kelly fraction from recent strategy performance
        recent_trades = portfolio.get_recent_trades(
            strategy=signal.metadata.get('strategy'),
            count=self.lookback_trades
        )
        
        if len(recent_trades) < 10:
            # Fall back to fixed percentage
            return int(portfolio.equity * 0.02 / market_data.current_bar()['close'].item())
        
        win_rate = sum(1 for t in recent_trades if t.pnl > 0) / len(recent_trades)
        avg_win = np.mean([t.pnl for t in recent_trades if t.pnl > 0])
        avg_loss = abs(np.mean([t.pnl for t in recent_trades if t.pnl <= 0]))
        
        if avg_loss == 0:
            kelly_f = 0.02  # Conservative fallback
        else:
            kelly_f = (win_rate - (1 - win_rate) * (avg_loss / avg_win)) / (avg_loss / avg_win)
            kelly_f = max(0, min(kelly_f, self.max_kelly_pct))
        
        current_price = market_data.current_bar()['close'].item()
        return int(portfolio.equity * kelly_f / current_price)

class VolatilityScaledSizer(PositionSizer):
    def __init__(self, target_vol: float = 0.02, lookback: int = 20):
        self.target_vol = target_vol
        self.lookback = lookback
    
    def calculate_size(self, signal: Signal, portfolio: Portfolio, 
                      market_data: MarketData) -> int:
        # Size position to target portfolio volatility
        recent_prices = market_data.historical(self.lookback)
        daily_returns = recent_prices['close'].pct_change().drop_nulls()
        current_vol = daily_returns.std()
        
        if current_vol <= 0:
            return 0
        
        vol_scalar = self.target_vol / current_vol
        current_price = market_data.current_bar()['close'].item()
        base_size = int(portfolio.equity * 0.02 / current_price)  # 2% base
        
        return int(base_size * vol_scalar)
```

## Risk Management as Middleware

### Composable Risk Components

```python
class RiskManager:
    """Orchestrates multiple risk checks."""
    
    def __init__(self):
        self.rules: list[RiskRule] = []
    
    def add_rule(self, rule: RiskRule) -> None:
        self.rules.append(rule)
    
    def evaluate(self, proposed_trade: ProposedTrade, 
                portfolio: Portfolio) -> RiskDecision:
        """Run all risk rules, return pass/block/modify decision."""
        for rule in self.rules:
            decision = rule.evaluate(proposed_trade, portfolio)
            if decision.action == "block":
                return decision
            elif decision.action == "modify":
                proposed_trade = decision.modified_trade
        
        return RiskDecision(action="pass", trade=proposed_trade)

class StopLossRule(RiskRule):
    def __init__(self, stop_pct: float):
        self.stop_pct = stop_pct
    
    def evaluate(self, trade: ProposedTrade, portfolio: Portfolio) -> RiskDecision:
        # Add stop-loss to trade metadata
        stop_price = trade.entry_price * (1 - self.stop_pct)
        trade.metadata["stop_loss"] = stop_price
        return RiskDecision(action="modify", trade=trade)

class PositionLimitRule(RiskRule):
    def __init__(self, max_position_pct: float):
        self.max_position_pct = max_position_pct
    
    def evaluate(self, trade: ProposedTrade, portfolio: Portfolio) -> RiskDecision:
        position_value = trade.entry_price * trade.shares
        position_pct = position_value / portfolio.equity
        
        if position_pct > self.max_position_pct:
            # Reduce position size
            max_shares = int(portfolio.equity * self.max_position_pct / trade.entry_price)
            if max_shares <= 0:
                return RiskDecision(action="block", reason="position_too_small")
            
            modified_trade = trade.copy()
            modified_trade.shares = max_shares
            return RiskDecision(action="modify", trade=modified_trade)
        
        return RiskDecision(action="pass", trade=trade)
```

## Strategy Registration & Versioning

### V1 Issues
- No versioning system
- Hard to track which strategy version produced which results
- No parameter change tracking

### V2 Versioning System

```python
from typing import Dict, Any
import hashlib
import yaml

class StrategyRegistry:
    """Central registry for all strategies with versioning."""
    
    def __init__(self):
        self._strategies: dict[str, dict[str, BaseStrategy]] = {}
    
    def register(self, strategy: BaseStrategy) -> None:
        """Register a strategy version."""
        if strategy.name not in self._strategies:
            self._strategies[strategy.name] = {}
        
        self._strategies[strategy.name][strategy.version] = strategy
    
    def get(self, name: str, version: str = "latest") -> BaseStrategy:
        """Get specific strategy version."""
        if version == "latest":
            versions = list(self._strategies[name].keys())
            version = max(versions, key=lambda v: self._parse_version(v))
        
        return self._strategies[name][version]
    
    def list_strategies(self) -> dict[str, list[str]]:
        """List all strategies and their versions."""
        return {name: list(versions.keys()) 
                for name, versions in self._strategies.items()}

@dataclass
class StrategyVersion:
    """Immutable strategy version metadata."""
    name: str
    version: str
    code_hash: str      # Hash of strategy code
    config_hash: str    # Hash of default configuration
    created_at: datetime
    author: str
    changelog: str
    
    @classmethod
    def from_strategy(cls, strategy: BaseStrategy, author: str, changelog: str):
        code_hash = cls._compute_code_hash(strategy)
        config_hash = cls._compute_config_hash(strategy.default_config)
        
        return cls(
            name=strategy.name,
            version=strategy.version,
            code_hash=code_hash,
            config_hash=config_hash,
            created_at=datetime.now(),
            author=author,
            changelog=changelog
        )

# Decorator for automatic registration
def strategy(name: str, version: str):
    def decorator(cls):
        cls.name = name
        cls.version = version
        STRATEGY_REGISTRY.register(cls())
        return cls
    return decorator
```

## Pine Script Conversion Framework

### Target: 90% Success Rate

```python
class PineScriptConverter:
    """Converts Pine Script strategies to V2 Python format."""
    
    def __init__(self):
        self.pine_functions = {
            'ta.sma': 'sma',
            'ta.ema': 'ema', 
            'ta.rsi': 'rsi',
            'ta.macd': 'macd',
            'ta.bb': 'bollinger_bands',
            # ... complete mapping
        }
    
    def convert(self, pine_code: str) -> tuple[str, float]:
        """Convert Pine Script to V2 Python strategy.
        
        Returns:
            (python_code, confidence_score)
        """
        # Parse Pine Script AST
        ast = self._parse_pine(pine_code)
        
        # Extract components
        inputs = self._extract_inputs(ast)
        indicators = self._extract_indicators(ast)
        signals = self._extract_signals(ast)
        
        # Generate Python code
        python_code = self._generate_strategy_class(
            name=self._extract_name(ast),
            inputs=inputs,
            indicators=indicators,
            signals=signals
        )
        
        # Compute confidence score based on supported features
        confidence = self._compute_confidence(ast)
        
        return python_code, confidence

# Generated strategy template
STRATEGY_TEMPLATE = '''
@strategy("{name}", "1.0.0")
class {class_name}(BaseStrategy):
    """Converted from Pine Script."""
    
    def __init__(self, {parameters}):
        {parameter_assignments}
    
    def get_indicators(self) -> list[IndicatorProvider]:
        return [
            {indicators}
        ]
    
    def get_signal_generator(self) -> SignalGenerator:
        return LambdaSignalGenerator(self._generate_signals)
    
    def _generate_signals(self, data: MarketData, indicators: pl.DataFrame) -> Signal:
        {signal_logic}
'''
```

## Event-Driven vs Vectorized Backtesting

### Hybrid Approach

V2 supports both paradigms depending on strategy needs:

```python
class BacktestMode(Enum):
    VECTORIZED = "vectorized"    # Fast, for simple strategies
    EVENT_DRIVEN = "event"       # Realistic, for complex strategies
    HYBRID = "hybrid"            # Vectorized indicators + event-driven signals

class StrategyExecutionEngine:
    """Executes strategies using the appropriate mode."""
    
    def run_vectorized(self, strategy: BaseStrategy, data: pl.DataFrame) -> BacktestResult:
        """Fast vectorized execution - all bars at once."""
        # Compute all indicators upfront
        indicators_df = self._compute_all_indicators(strategy, data)
        
        # Generate all signals vectorized
        signals_df = self._generate_all_signals(strategy, data, indicators_df)
        
        # Simulate trades vectorized
        trades = self._simulate_trades_vectorized(signals_df, data)
        
        return BacktestResult(trades=trades, equity_curve=equity_curve)
    
    def run_event_driven(self, strategy: BaseStrategy, data: pl.DataFrame) -> BacktestResult:
        """Realistic event-driven execution - bar by bar."""
        portfolio = Portfolio(initial_capital=self.initial_capital)
        trades = []
        
        for bar_idx in range(strategy.get_warmup_periods(), len(data)):
            market_data = MarketData(
                primary_df=data, 
                secondary_dfs=self.secondary_data,
                current_idx=bar_idx,
                symbol=symbol
            )
            
            # Process this bar
            signal = strategy.generate_signal(market_data)
            if signal.action != "hold":
                proposed_trade = self.position_sizer.size_position(signal, portfolio)
                risk_decision = self.risk_manager.evaluate(proposed_trade, portfolio)
                
                if risk_decision.action == "pass":
                    executed_trade = self.execution_engine.execute(
                        risk_decision.trade, market_data, portfolio
                    )
                    trades.append(executed_trade)
                    portfolio.update(executed_trade)
        
        return BacktestResult(trades=trades, portfolio=portfolio)
```

## Strategy Testing Framework

### Unit Testing Made Easy

```python
import pytest
from v2.testing import StrategyTester, MockMarketData

class TestBollingerStrategy:
    """Comprehensive strategy testing."""
    
    @pytest.fixture
    def strategy(self):
        return BollingerMeanReversion(bb_period=20, bb_std=2.0)
    
    @pytest.fixture 
    def market_data(self):
        # Generate realistic test data
        return MockMarketData.trending_up(bars=100, volatility=0.02)
    
    def test_no_lookahead_bias(self, strategy, market_data):
        """Verify strategy cannot access future data."""
        tester = StrategyTester(strategy)
        
        # This should pass without exception
        result = tester.test_temporal_constraints(market_data)
        assert result.lookahead_violations == 0
    
    def test_signal_generation_deterministic(self, strategy, market_data):
        """Same inputs should produce same outputs."""
        tester = StrategyTester(strategy)
        
        result1 = tester.run_backtest(market_data, seed=42)
        result2 = tester.run_backtest(market_data, seed=42)
        
        assert result1.trades == result2.trades
        assert result1.equity_curve.equals(result2.equity_curve)
    
    def test_edge_cases(self, strategy):
        """Test strategy with problematic data."""
        # Missing data
        gappy_data = MockMarketData.with_gaps(missing_pct=0.1)
        result = StrategyTester(strategy).run_backtest(gappy_data)
        assert result.is_valid()
        
        # Extreme volatility
        volatile_data = MockMarketData.high_volatility(volatility=0.10)
        result = StrategyTester(strategy).run_backtest(volatile_data)
        assert not result.has_excessive_losses()
        
        # Low liquidity  
        illiquid_data = MockMarketData.low_volume(avg_volume=1000)
        result = StrategyTester(strategy).run_backtest(illiquid_data)
        assert result.respects_volume_constraints()
```

## Configuration Evolution

### V2 Enhanced Config Format

```yaml
# V2 configuration - more structured and declarative
meta:
  strategy: bollinger_mean_reversion
  version: "2.1.0"
  description: "BB mean reversion with RSI confirmation and regime filtering"
  tags: ["mean_reversion", "bollinger_bands", "multi_timeframe"]
  
runtime:
  primary_timeframe: "15m"
  required_timeframes: ["1d", "4h", "15m"]
  warmup_periods: 50
  execution_mode: "event_driven"  # vectorized, event_driven, hybrid
  
data:
  symbols: # Override universe
    - AAPL
    - MSFT
    - GOOGL
  universe: "sp500_liquid"  # Or use predefined universe
  date_range:
    start: "2020-01-01"
    end: "2024-12-31"
    
components:
  indicators:
    trend_ema:
      type: "exponential_ma"
      timeframe: "1d"
      params: {period: 50}
      
    bb_main:
      type: "bollinger_bands" 
      timeframe: "15m"
      params: {period: 20, std_dev: 2.0}
      
    rsi_confirm:
      type: "rsi"
      timeframe: "15m" 
      params: {period: 14}
  
  signals:
    entry_long:
      conditions:
        - bb_touch_lower: {indicator: "bb_main"}
        - rsi_oversold: {indicator: "rsi_confirm", threshold: 30}
        - trend_bullish: {indicator: "trend_ema", method: "price_above"}
      logic: "all"  # all, any, weighted
      
    exit_long:
      conditions:
        - bb_touch_upper: {indicator: "bb_main"}
        - rsi_overbought: {indicator: "rsi_confirm", threshold: 70}
      logic: "any"
      
  sizing:
    type: "kelly_optimal"
    params:
      lookback_trades: 50
      max_kelly_pct: 0.25
      min_position_dollars: 1000
      max_position_pct: 0.05
      
  risk:
    rules:
      - type: "trailing_stop"
        params: {pct: 0.02, activation_pct: 0.01}
      - type: "max_exposure"
        params: {total_pct: 0.8, per_sector_pct: 0.3}
      - type: "correlation_limit"
        params: {max_correlation: 0.7, lookback_days: 30}
        
  regime:
    enabled: true
    allowed_regimes: ["bull", "neutral"]
    block_regimes: ["bear", "crisis"]
    confidence_threshold: 0.8
    
execution:
  initial_capital: 100000.0
  fill_model: "realistic_volume"  # market, limit, realistic_volume
  commission_model: "interactive_brokers"
  slippage_model: "sqrt_volume"
  max_workers: 8
  
optimization:
  enabled: false
  method: "bayesian"  # grid, random, bayesian, genetic
  objective: "sharpe_ratio"  # sharpe_ratio, calmar_ratio, sortino_ratio
  trials: 100
  cv_folds: 5  # walk-forward cross-validation
  
output:
  save_trades: true
  save_signals: false  # Can generate massive data
  save_equity_curve: true
  save_attribution: true
  export_format: ["json", "csv"]
```

## Strategy Builder Pattern

### Fluent Interface for Complex Strategies

```python
from v2.core.strategy import StrategyBuilder

# Build strategies programmatically
def create_sophisticated_strategy():
    return StrategyBuilder("multi_factor_momentum") \
        .version("1.0.0") \
        .description("Multi-factor momentum with regime awareness") \
        .primary_timeframe("1d") \
        .add_timeframe("1w") \
        .add_timeframe("1h") \
        \
        .add_indicator(RSI(period=14, timeframe="1d")) \
        .add_indicator(MACD(fast=12, slow=26, signal=9, timeframe="1d")) \
        .add_indicator(BollingerBands(period=20, std_dev=2, timeframe="1h")) \
        .add_indicator(VolumeMA(period=20, timeframe="1h")) \
        \
        .entry_condition("momentum_up", 
                        lambda data, ind: (ind['rsi'] > 50) & (ind['macd_signal'] > 0)) \
        .entry_condition("volume_confirm",
                        lambda data, ind: data.current('volume') > ind['volume_ma'] * 1.5) \
        .entry_logic("all") \
        \
        .exit_condition("momentum_down",
                       lambda data, ind: ind['rsi'] < 40) \
        .exit_condition("bb_upper_touch", 
                       lambda data, ind: data.current('close') > ind['bb_upper']) \
        .exit_logic("any") \
        \
        .position_sizer(KellySizer(lookback=50, max_kelly=0.2)) \
        .risk_manager(RiskManager() \
                     .add_rule(TrailingStop(pct=0.02)) \
                     .add_rule(MaxDrawdown(pct=0.15))) \
        \
        .regime_filter(enabled=True, allowed=["bull", "neutral"]) \
        \
        .build()
```

## Performance Optimizations

### Indicator Caching
```python
class CachedIndicatorProvider:
    """Caches computed indicators to avoid recomputation."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.local_cache = {}
    
    def get_or_compute(self, indicator: IndicatorProvider, 
                      data: pl.DataFrame) -> pl.DataFrame:
        # Create cache key from indicator config + data hash
        data_hash = hashlib.sha256(data.hash_rows().to_pandas().values.tobytes()).hexdigest()[:16]
        cache_key = f"indicator:{indicator.name}:{indicator.config_hash}:{data_hash}"
        
        # Check local cache first
        if cache_key in self.local_cache:
            return self.local_cache[cache_key]
        
        # Check Redis
        cached_result = self.redis.get(cache_key)
        if cached_result:
            result = pl.read_parquet(BytesIO(cached_result))
            self.local_cache[cache_key] = result
            return result
        
        # Compute and cache
        result = indicator.compute(data)
        self.redis.setex(cache_key, 3600, result.write_parquet().getvalue())
        self.local_cache[cache_key] = result
        return result
```

### Parallel Strategy Execution
```python
class ParallelStrategyRunner:
    """Execute multiple strategies in parallel."""
    
    async def run_strategies(self, strategies: list[BaseStrategy], 
                           symbols: list[str]) -> dict[str, BacktestResult]:
        # Create work units: (strategy, symbol) pairs
        work_units = [(s, sym) for s in strategies for sym in symbols]
        
        # Process in batches with semaphore
        semaphore = asyncio.Semaphore(self.max_workers)
        results = {}
        
        async def _run_unit(strategy: BaseStrategy, symbol: str):
            async with semaphore:
                result = await self._run_single_strategy(strategy, symbol)
                return f"{strategy.name}:{symbol}", result
        
        # Execute all work units
        tasks = [_run_unit(s, sym) for s, sym in work_units]
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for item in completed:
            if isinstance(item, Exception):
                logger.error("Strategy execution failed: %s", item)
                continue
            key, result = item
            results[key] = result
        
        return results
```

## Migration from V1

### Automated Migration Tool
```python
class V1ToV2Migrator:
    """Migrate V1 strategies to V2 format."""
    
    def migrate_strategy(self, v1_strategy_path: str) -> str:
        """Convert V1 strategy to V2 format."""
        v1_code = Path(v1_strategy_path).read_text()
        
        # Parse V1 strategy
        v1_ast = ast.parse(v1_code)
        class_node = self._find_strategy_class(v1_ast)
        
        # Extract components
        indicators = self._extract_v1_indicators(class_node)
        signals = self._extract_v1_signals(class_node)
        trades = self._extract_v1_trades(class_node)
        
        # Generate V2 equivalent
        v2_code = self._generate_v2_strategy(
            name=class_node.name,
            indicators=indicators,
            signals=signals,
            execution=trades
        )
        
        return v2_code
    
    def validate_migration(self, v1_strategy: str, v2_strategy: str, 
                          test_data: pl.DataFrame) -> MigrationReport:
        """Validate that V2 produces same results as V1."""
        v1_result = self._run_v1_strategy(v1_strategy, test_data)
        v2_result = self._run_v2_strategy(v2_strategy, test_data)
        
        return MigrationReport(
            trades_match=self._compare_trades(v1_result.trades, v2_result.trades),
            signals_match=self._compare_signals(v1_result.signals, v2_result.signals),
            pnl_difference=abs(v1_result.total_pnl - v2_result.total_pnl),
            confidence_score=self._compute_migration_confidence()
        )
```

## Strategy Composition Examples

### Simple Strategy
```python
@strategy("simple_ma_cross", "2.0.0")
class SimpleMAStrategy(BaseStrategy):
    def __init__(self, fast_period=9, slow_period=21):
        self.fast_ma = EMA(period=fast_period)
        self.slow_ma = EMA(period=slow_period)
    
    def get_indicators(self):
        return [self.fast_ma, self.slow_ma]
    
    def get_signal_generator(self):
        return CrossoverSignalGenerator(
            fast_line="ema_fast",
            slow_line="ema_slow"
        )
```

### Complex Multi-Factor Strategy
```python
@strategy("multi_factor_momentum", "2.0.0") 
class MultiFactor(BaseStrategy):
    def __init__(self):
        # Multiple timeframes
        self.daily_trend = EMA(period=50, timeframe="1d")
        self.hourly_momentum = RSI(period=14, timeframe="1h")
        self.minute_entry = BollingerBands(period=20, timeframe="15m")
        
        # Complex signal logic
        self.signal_gen = WeightedSignalGenerator([
            (TrendSignal(self.daily_trend), 0.4),
            (MomentumSignal(self.hourly_momentum), 0.3),
            (MeanReversionSignal(self.minute_entry), 0.3)
        ])
    
    def get_indicators(self):
        return [self.daily_trend, self.hourly_momentum, self.minute_entry]
    
    def get_signal_generator(self):
        return self.signal_gen
    
    def get_required_timeframes(self):
        return ["1d", "1h", "15m"]
```

## Benefits Over V1

### Developer Experience
- **10x less code** for common strategies
- **Zero boilerplate** for standard patterns  
- **Instant feedback** via comprehensive testing
- **Pine Script import** reduces strategy development time
- **Component reuse** across strategies

### Performance
- **Vectorized indicators** computed once, reused across strategies
- **Smart caching** avoids redundant computation
- **Parallel execution** across symbols and strategies
- **Memory efficiency** through lazy loading

### Correctness
- **No lookahead bias possible** — enforced by data flow architecture
- **Deterministic results** — same inputs always produce same outputs
- **Comprehensive testing** — every component tested in isolation
- **Attribution tracking** — understand why every trade was made

### Flexibility
- **Mix and match** sizing and risk components
- **A/B testing** of strategy variants
- **Parameter optimization** with overfitting protection
- **Walk-forward analysis** for robust validation

This V2 strategy framework transforms strategy development from art to science while maintaining the flexibility that makes backtesting powerful.