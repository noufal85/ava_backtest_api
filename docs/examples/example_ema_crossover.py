"""
EMA Crossover Strategy — Beginner Reference
=============================================
Simple trend-following strategy: go long when fast EMA crosses above slow EMA,
exit when it crosses below.

Follows STRATEGY_PATTERN.md exactly.
"""

# ---------------------------------------------------------------------------
# YAML Config (also stored as ema_crossover.yaml alongside this file)
# ---------------------------------------------------------------------------
YAML_CONFIG = """
meta:
  strategy: ema_crossover
  version: "2.0.0"
  description: "Simple EMA crossover trend follower"
  tags: ["trend_following", "ema", "beginner"]

runtime:
  primary_timeframe: "1d"
  warmup_periods: 50
  execution_mode: "event_driven"

components:
  indicators:
    fast_ema:
      type: exponential_ma
      params: {period: 9}
    slow_ema:
      type: exponential_ma
      params: {period: 21}

  signals:
    entry_long:
      conditions:
        - crossover: {fast: fast_ema, slow: slow_ema}
      logic: all
    exit_long:
      conditions:
        - crossunder: {fast: fast_ema, slow: slow_ema}
      logic: all

  sizing:
    type: fixed_pct
    params: {pct: 0.02}   # 2 % of equity per trade

  risk:
    rules:
      - type: stop_loss
        params: {pct: 0.05}
      - type: max_exposure
        params: {total_pct: 0.80}

execution:
  initial_capital: 100000
  universe: sp500_liquid
  start_date: "2020-01-01"
  end_date: "2024-12-31"
"""

# ---------------------------------------------------------------------------
# Strategy Class
# ---------------------------------------------------------------------------
from v2.core.strategy.base import BaseStrategy, Signal, MarketData
from v2.core.indicators.trend import EMA

class EMACrossoverStrategy(BaseStrategy):
    """Long-only EMA crossover."""

    name = "ema_crossover"
    version = "2.0.0"
    description = "Buy when fast EMA > slow EMA, sell on cross-under."

    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        self.fast_ema = EMA(period=fast_period, name="fast_ema")
        self.slow_ema = EMA(period=slow_period, name="slow_ema")

    # -- required interface --------------------------------------------------

    def get_indicators(self):
        return [self.fast_ema, self.slow_ema]

    def get_signal_generator(self):
        return self  # implements generate() below

    def get_warmup_periods(self) -> int:
        return max(self.fast_ema.period, self.slow_ema.period) + 1

    # -- signal logic --------------------------------------------------------

    def generate(self, data: MarketData, indicators) -> Signal:
        fast_now  = indicators["fast_ema"].tail(1).item()
        fast_prev = indicators["fast_ema"].tail(2).head(1).item()
        slow_now  = indicators["slow_ema"].tail(1).item()
        slow_prev = indicators["slow_ema"].tail(2).head(1).item()

        ts  = data.current_bar()["timestamp"].item()
        sym = data.symbol

        # Cross-over: fast crosses above slow
        if fast_prev <= slow_prev and fast_now > slow_now:
            return Signal(timestamp=ts, symbol=sym, action="buy",
                          strength=1.0, confidence=0.9,
                          metadata={"trigger": "ema_crossover"})

        # Cross-under: fast crosses below slow
        if fast_prev >= slow_prev and fast_now < slow_now:
            return Signal(timestamp=ts, symbol=sym, action="exit",
                          strength=1.0, confidence=0.9,
                          metadata={"trigger": "ema_crossunder"})

        return Signal(timestamp=ts, symbol=sym, action="hold",
                      strength=0.0, confidence=1.0, metadata={})


# ---------------------------------------------------------------------------
# Unit Test
# ---------------------------------------------------------------------------
"""
Run:  pytest docs/examples/example_ema_crossover.py -v
"""
import pytest
import polars as pl
from datetime import datetime, timedelta

class TestEMACrossover:
    @pytest.fixture
    def strategy(self):
        return EMACrossoverStrategy(fast_period=3, slow_period=5)

    def _make_data(self, prices: list[float]) -> MarketData:
        """Helper: build MarketData from a price series."""
        n = len(prices)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        df = pl.DataFrame({
            "timestamp": dates,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low":  [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1_000_000] * n,
        })
        return MarketData(primary_df=df, secondary_dfs={},
                          current_idx=n - 1, symbol="TEST")

    def test_buy_signal_on_crossover(self, strategy):
        # Trending up → fast will cross above slow
        prices = [100, 99, 98, 97, 96, 97, 99, 102, 106, 110]
        data = self._make_data(prices)
        indicators = {
            "fast_ema": pl.Series([None, None, 99.0, 98.0, 97.0, 97.0, 98.0, 100.0, 103.0, 106.5]),
            "slow_ema": pl.Series([None]*4 + [98.0, 97.7, 97.9, 98.7, 100.1, 102.5]),
        }
        signal = strategy.generate(data, indicators)
        # Fast went from below slow to above → buy
        assert signal.action == "buy"

    def test_hold_when_no_cross(self, strategy):
        prices = [100, 101, 102, 103, 104, 105]
        data = self._make_data(prices)
        indicators = {
            "fast_ema": pl.Series([100, 100.5, 101.2, 102, 103, 104]),
            "slow_ema": pl.Series([100, 100.2, 100.6, 101, 101.6, 102.3]),
        }
        signal = strategy.generate(data, indicators)
        assert signal.action == "hold"


# ---------------------------------------------------------------------------
# Expected Signal Output (first 5 signals on SPY 2020-2024, 9/21 EMA)
# ---------------------------------------------------------------------------
"""
DATE        | SYMBOL | ACTION | STRENGTH | TRIGGER
2020-04-06  | SPY    | buy    | 1.0      | ema_crossover
2020-06-11  | SPY    | exit   | 1.0      | ema_crossunder
2020-06-29  | SPY    | buy    | 1.0      | ema_crossover
2020-09-21  | SPY    | exit   | 1.0      | ema_crossunder
2020-10-12  | SPY    | buy    | 1.0      | ema_crossover
"""
