"""
Bollinger Band Mean Reversion — Intermediate Reference
=======================================================
Buy when price touches/pierces the lower Bollinger Band with RSI confirmation;
exit at the middle band or upper band.
"""

YAML_CONFIG = """
meta:
  strategy: bb_mean_reversion
  version: "2.0.0"
  description: "Bollinger Band mean reversion with RSI filter"
  tags: ["mean_reversion", "bollinger", "intermediate"]

runtime:
  primary_timeframe: "1d"
  warmup_periods: 30
  execution_mode: "event_driven"

components:
  indicators:
    bb:
      type: bollinger_bands
      params: {period: 20, std_dev: 2.0}
    rsi:
      type: rsi
      params: {period: 14}

  signals:
    entry_long:
      conditions:
        - price_below: {ref: bb.lower}
        - rsi_below: {ref: rsi, threshold: 35}
      logic: all
    exit_long:
      conditions:
        - price_above: {ref: bb.middle}
      logic: any

  sizing:
    type: volatility_scaled
    params: {target_vol: 0.02, lookback: 20}

  risk:
    rules:
      - type: stop_loss
        params: {pct: 0.04}
      - type: max_position
        params: {pct: 0.05}

execution:
  initial_capital: 100000
  universe: sp500_liquid
  start_date: "2020-01-01"
  end_date: "2024-12-31"
"""

# ---------------------------------------------------------------------------
from v2.core.strategy.base import BaseStrategy, Signal, MarketData
from v2.core.indicators.volatility import BollingerBands
from v2.core.indicators.momentum import RSI


class BBMeanReversionStrategy(BaseStrategy):
    name = "bb_mean_reversion"
    version = "2.0.0"
    description = "Bollinger Band mean reversion with RSI confirmation"

    def __init__(self, bb_period=20, bb_std=2.0, rsi_period=14,
                 rsi_oversold=35, exit_at="middle"):
        self.bb = BollingerBands(period=bb_period, std_dev=bb_std, name="bb")
        self.rsi = RSI(period=rsi_period, name="rsi")
        self.rsi_oversold = rsi_oversold
        self.exit_at = exit_at  # "middle" or "upper"

    def get_indicators(self):
        return [self.bb, self.rsi]

    def get_signal_generator(self):
        return self

    def get_warmup_periods(self) -> int:
        return max(self.bb.period, self.rsi.period) + 5

    def generate(self, data: MarketData, indicators) -> Signal:
        bar = data.current_bar()
        close = bar["close"].item()
        low = bar["low"].item()
        ts = bar["timestamp"].item()
        sym = data.symbol

        bb_lower  = indicators["bb_lower"].tail(1).item()
        bb_middle = indicators["bb_middle"].tail(1).item()
        bb_upper  = indicators["bb_upper"].tail(1).item()
        rsi_val   = indicators["rsi"].tail(1).item()

        # Entry: price touches lower band + RSI oversold
        if low <= bb_lower and rsi_val < self.rsi_oversold:
            strength = min(1.0, (bb_lower - low) / bb_lower * 100)
            return Signal(timestamp=ts, symbol=sym, action="buy",
                          strength=strength, confidence=0.85,
                          metadata={"trigger": "bb_lower_rsi",
                                    "bb_lower": bb_lower, "rsi": rsi_val})

        # Exit: price reaches target band
        target = bb_middle if self.exit_at == "middle" else bb_upper
        if close >= target:
            return Signal(timestamp=ts, symbol=sym, action="exit",
                          strength=1.0, confidence=0.9,
                          metadata={"trigger": f"bb_{self.exit_at}_touch"})

        return Signal(timestamp=ts, symbol=sym, action="hold",
                      strength=0.0, confidence=1.0, metadata={})


# ---------------------------------------------------------------------------
# Unit Test
# ---------------------------------------------------------------------------
import pytest, polars as pl
from datetime import datetime, timedelta

class TestBBMeanReversion:
    @pytest.fixture
    def strategy(self):
        return BBMeanReversionStrategy(bb_period=20, bb_std=2.0,
                                       rsi_period=14, rsi_oversold=35)

    def _bar(self, close, low=None):
        low = low or close
        return pl.DataFrame({
            "timestamp": [datetime(2024, 6, 1)],
            "open": [close], "high": [close * 1.01],
            "low": [low], "close": [close], "volume": [1_000_000],
        })

    def test_entry_on_bb_lower_touch(self, strategy):
        data = MarketData(primary_df=self._bar(95, low=94),
                          secondary_dfs={}, current_idx=0, symbol="TEST")
        indicators = {
            "bb_lower": pl.Series([95.0]),
            "bb_middle": pl.Series([100.0]),
            "bb_upper": pl.Series([105.0]),
            "rsi": pl.Series([28.0]),
        }
        sig = strategy.generate(data, indicators)
        assert sig.action == "buy"

    def test_no_entry_when_rsi_high(self, strategy):
        data = MarketData(primary_df=self._bar(95, low=94),
                          secondary_dfs={}, current_idx=0, symbol="TEST")
        indicators = {
            "bb_lower": pl.Series([95.0]),
            "bb_middle": pl.Series([100.0]),
            "bb_upper": pl.Series([105.0]),
            "rsi": pl.Series([55.0]),
        }
        sig = strategy.generate(data, indicators)
        assert sig.action == "hold"

    def test_exit_at_middle_band(self, strategy):
        data = MarketData(primary_df=self._bar(100.5),
                          secondary_dfs={}, current_idx=0, symbol="TEST")
        indicators = {
            "bb_lower": pl.Series([95.0]),
            "bb_middle": pl.Series([100.0]),
            "bb_upper": pl.Series([105.0]),
            "rsi": pl.Series([55.0]),
        }
        sig = strategy.generate(data, indicators)
        assert sig.action == "exit"


# ---------------------------------------------------------------------------
# Expected Signal Output (AAPL, 2023, BB(20,2) + RSI(14)<35)
# ---------------------------------------------------------------------------
"""
DATE        | SYMBOL | ACTION | RSI  | BB_LOWER | TRIGGER
2023-01-05  | AAPL   | buy    | 32.1 | 127.50   | bb_lower_rsi
2023-01-18  | AAPL   | exit   | 52.3 | —        | bb_middle_touch
2023-03-13  | AAPL   | buy    | 29.8 | 146.20   | bb_lower_rsi
2023-03-27  | AAPL   | exit   | 58.1 | —        | bb_middle_touch
"""
