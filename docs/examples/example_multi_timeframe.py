"""
Multi-Timeframe Momentum — Advanced Reference
===============================================
Uses daily EMA for trend direction, hourly RSI for momentum,
and 15-minute Bollinger Bands for precise entry timing.

Demonstrates native multi-timeframe support per STRATEGY_PATTERN.md.
"""

YAML_CONFIG = """
meta:
  strategy: multi_tf_momentum
  version: "2.0.0"
  description: "Multi-timeframe momentum: daily trend + hourly momentum + 15m entry"
  tags: ["multi_timeframe", "momentum", "advanced"]

runtime:
  primary_timeframe: "15m"
  required_timeframes: ["1d", "1h", "15m"]
  warmup_periods: 100
  execution_mode: "event_driven"

components:
  indicators:
    daily_ema:
      type: exponential_ma
      timeframe: "1d"
      params: {period: 50}
    hourly_rsi:
      type: rsi
      timeframe: "1h"
      params: {period: 14}
    entry_bb:
      type: bollinger_bands
      timeframe: "15m"
      params: {period: 20, std_dev: 2.0}

  signals:
    entry_long:
      conditions:
        - trend_bullish: {indicator: daily_ema, method: price_above}
        - momentum_rising: {indicator: hourly_rsi, min: 40, max: 70}
        - bb_lower_touch: {indicator: entry_bb}
      logic: all
    exit_long:
      conditions:
        - trend_bearish: {indicator: daily_ema, method: price_below}
        - bb_upper_touch: {indicator: entry_bb}
      logic: any

  sizing:
    type: kelly_optimal
    params: {lookback_trades: 50, max_kelly_pct: 0.15}

  risk:
    rules:
      - type: trailing_stop
        params: {pct: 0.02, activation_pct: 0.01}
      - type: max_exposure
        params: {total_pct: 0.60, per_sector_pct: 0.20}
      - type: correlation_limit
        params: {max_correlation: 0.7, lookback_days: 30}

  regime:
    enabled: true
    allowed_regimes: ["bull", "neutral"]
    confidence_threshold: 0.8

execution:
  initial_capital: 100000
  universe: sp500_liquid
  start_date: "2022-01-01"
  end_date: "2024-12-31"
"""

# ---------------------------------------------------------------------------
from v2.core.strategy.base import BaseStrategy, Signal, MarketData
from v2.core.indicators.trend import EMA
from v2.core.indicators.momentum import RSI
from v2.core.indicators.volatility import BollingerBands


class MultiTimeframeMomentum(BaseStrategy):
    name = "multi_tf_momentum"
    version = "2.0.0"
    description = "Daily trend + hourly RSI + 15m BB entry"

    def __init__(self, ema_period=50, rsi_period=14,
                 bb_period=20, bb_std=2.0,
                 rsi_low=40, rsi_high=70):
        self.daily_ema = EMA(period=ema_period, timeframe="1d", name="daily_ema")
        self.hourly_rsi = RSI(period=rsi_period, timeframe="1h", name="hourly_rsi")
        self.entry_bb = BollingerBands(period=bb_period, std_dev=bb_std,
                                        timeframe="15m", name="entry_bb")
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high

    def get_indicators(self):
        return [self.daily_ema, self.hourly_rsi, self.entry_bb]

    def get_required_timeframes(self):
        return ["1d", "1h", "15m"]

    def get_signal_generator(self):
        return self

    def get_warmup_periods(self) -> int:
        return 100  # need 50 daily bars aligned to 15m grid

    def generate(self, data: MarketData, indicators) -> Signal:
        bar = data.current_bar()
        ts = bar["timestamp"].item()
        sym = data.symbol
        close = bar["close"].item()
        low = bar["low"].item()

        # --- Daily trend filter ---
        daily_ema_val = indicators["daily_ema"].tail(1).item()
        trend_bullish = close > daily_ema_val

        # --- Hourly momentum ---
        rsi_val = indicators["hourly_rsi"].tail(1).item()
        momentum_ok = self.rsi_low <= rsi_val <= self.rsi_high

        # --- 15m BB entry ---
        bb_lower = indicators["entry_bb_lower"].tail(1).item()
        bb_upper = indicators["entry_bb_upper"].tail(1).item()

        # ENTRY: all three conditions met
        if trend_bullish and momentum_ok and low <= bb_lower:
            return Signal(
                timestamp=ts, symbol=sym, action="buy",
                strength=min(1.0, (bb_lower - low) / bb_lower * 100),
                confidence=0.75,
                metadata={"trigger": "multi_tf_entry",
                          "daily_ema": daily_ema_val,
                          "hourly_rsi": rsi_val,
                          "bb_lower": bb_lower})

        # EXIT: trend flips OR upper band hit
        if not trend_bullish:
            return Signal(
                timestamp=ts, symbol=sym, action="exit",
                strength=1.0, confidence=0.9,
                metadata={"trigger": "trend_bearish"})

        if close >= bb_upper:
            return Signal(
                timestamp=ts, symbol=sym, action="exit",
                strength=0.8, confidence=0.85,
                metadata={"trigger": "bb_upper_touch"})

        return Signal(timestamp=ts, symbol=sym, action="hold",
                      strength=0.0, confidence=1.0, metadata={})


# ---------------------------------------------------------------------------
# Unit Test
# ---------------------------------------------------------------------------
import pytest, polars as pl
from datetime import datetime

class TestMultiTimeframe:
    @pytest.fixture
    def strategy(self):
        return MultiTimeframeMomentum(ema_period=50, rsi_period=14,
                                      bb_period=20, bb_std=2.0)

    def _bar(self, close, low=None):
        low = low or close
        return pl.DataFrame({
            "timestamp": [datetime(2024, 6, 1, 10, 30)],
            "open": [close], "high": [close * 1.01],
            "low": [low], "close": [close], "volume": [500_000],
        })

    def test_entry_all_conditions_met(self, strategy):
        data = MarketData(primary_df=self._bar(152, low=149),
                          secondary_dfs={}, current_idx=0, symbol="AAPL")
        indicators = {
            "daily_ema": pl.Series([148.0]),      # close > ema → bullish
            "hourly_rsi": pl.Series([55.0]),       # 40 ≤ rsi ≤ 70
            "entry_bb_lower": pl.Series([150.0]),  # low ≤ bb_lower
            "entry_bb_upper": pl.Series([158.0]),
        }
        sig = strategy.generate(data, indicators)
        assert sig.action == "buy"

    def test_no_entry_bearish_trend(self, strategy):
        data = MarketData(primary_df=self._bar(145, low=143),
                          secondary_dfs={}, current_idx=0, symbol="AAPL")
        indicators = {
            "daily_ema": pl.Series([148.0]),      # close < ema → bearish
            "hourly_rsi": pl.Series([55.0]),
            "entry_bb_lower": pl.Series([144.0]),
            "entry_bb_upper": pl.Series([152.0]),
        }
        sig = strategy.generate(data, indicators)
        assert sig.action == "exit"  # exits due to bearish trend

    def test_exit_on_upper_band(self, strategy):
        data = MarketData(primary_df=self._bar(159),
                          secondary_dfs={}, current_idx=0, symbol="AAPL")
        indicators = {
            "daily_ema": pl.Series([148.0]),
            "hourly_rsi": pl.Series([68.0]),
            "entry_bb_lower": pl.Series([150.0]),
            "entry_bb_upper": pl.Series([158.0]),
        }
        sig = strategy.generate(data, indicators)
        assert sig.action == "exit"


# ---------------------------------------------------------------------------
# Expected Signal Output (AAPL, 15m bars, 2024-Q1 excerpt)
# ---------------------------------------------------------------------------
"""
TIMESTAMP            | SYMBOL | ACTION | TRIGGER         | DAILY_EMA | H_RSI | BB_LOWER
2024-01-08 10:45:00  | AAPL   | buy    | multi_tf_entry  | 185.30    | 48.2  | 183.00
2024-01-09 14:00:00  | AAPL   | exit   | bb_upper_touch  | 185.30    | 65.1  | —
2024-01-22 11:15:00  | AAPL   | buy    | multi_tf_entry  | 186.10    | 42.5  | 184.50
2024-01-24 15:30:00  | AAPL   | exit   | trend_bearish   | 186.50    | 38.0  | —
"""
