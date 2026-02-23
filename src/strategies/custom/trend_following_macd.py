from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import polars as pl
import numpy as np
import math

from src.core.strategy.registry import register

@dataclass
class Signal:
    action: str          # "buy" | "sell" | "hold"
    strength: float      # 0.0 – 1.0
    confidence: float    # 0.0 – 1.0
    metadata: dict = field(default_factory=dict)

class BaseStrategy(ABC):
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    category: str = ""   # trend | mean_reversion | momentum | multi_factor | volatility
    tags: list[str] = []

    @abstractmethod
    def get_warmup_periods(self) -> int: ...

    @abstractmethod
    def generate_signal(self, window) -> Signal | None:
        # window.current_bar()   → polars DataFrame row (current bar)
        # window.historical()    → polars DataFrame of all historical bars
        # Columns: open, high, low, close, volume (float64), timestamp (date)
        # Combine: pl.concat([window.historical(), window.current_bar()])
        # NEVER look ahead
        ...

    def get_parameter_schema(self) -> dict:
        return {}

def ema(series: pl.Series, period: int) -> pl.Series:
    alpha = 2 / (period + 1)
    ema = [series[0]]
    for i in range(1, len(series)):
        ema.append(alpha * series[i] + (1 - alpha) * ema[-1])
    return pl.Series(ema)

def macd(series: pl.Series, fast: int, slow: int, signal: int) -> tuple[pl.Series, pl.Series, pl.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def atr(df: pl.DataFrame, period: int) -> pl.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr = [high[0] - low[0]]
    for i in range(1, len(df)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    tr_series = pl.Series(tr)

    atr_values = [tr_series[:period].mean()]
    for i in range(period, len(df)):
        atr_values.append((atr_values[-1] * (period - 1) + tr_series[i]) / period)
    return pl.Series(atr_values)

@register
class TrendFollowingMacd(BaseStrategy):
    name = "trend_following_macd"
    version = "1.0.0"
    description = "Trend following with 20/50 MA crossover + MACD confirmation"
    category = "trend"
    tags = ["trend following", "macd", "ma crossover"]

    def __init__(self,
                 ma_fast: int = 20,
                 ma_slow: int = 50,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 atr_period: int = 14,
                 atr_stop_mult: float = 2.5,
                 max_hold: int = 40):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.max_hold = max_hold
        self.hold_count = 0
        self.in_trade = False
        self.highest_price = 0.0

    def get_warmup_periods(self) -> int:
        return max(self.ma_fast, self.ma_slow, self.macd_fast, self.macd_slow, self.macd_signal, self.atr_period) + 5

    def generate_signal(self, window) -> Signal | None:
        df = window.historical()
        if len(df) < self.get_warmup_periods():
            return None

        closes = df["close"]
        highs = df["high"]
        lows = df["low"]

        ma_fast_values = ema(closes, self.ma_fast)
        ma_slow_values = ema(closes, self.ma_slow)
        macd_line, signal_line, _ = macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        atr_values = atr(df, self.atr_period)

        ma_fast = ma_fast_values[-1]
        ma_slow = ma_slow_values[-1]
        macd_val = macd_line[-1]
        signal_val = signal_line[-1]
        current_atr = atr_values[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        current_close = closes[-1]

        if not self.in_trade:
            if ma_fast > ma_slow and macd_val > signal_val:
                self.in_trade = True
                self.hold_count = 1
                self.highest_price = current_high
                return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"reason": "ma_macd_confirmation"})
            else:
                return None
        else:
            self.hold_count += 1
            self.highest_price = max(self.highest_price, current_high)
            trailing_stop = self.highest_price - self.atr_stop_mult * current_atr
            stop_hit = current_low <= trailing_stop and current_atr > 0
            ma_cross_down = ma_fast < ma_slow
            macd_bearish = macd_val < signal_val
            max_hold_reached = self.hold_count > self.max_hold

            if stop_hit or ma_cross_down or macd_bearish or max_hold_reached:
                self.in_trade = False
                self.hold_count = 0
                self.highest_price = 0.0
                return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"reason": "exit_condition"})
            else:
                return None

    def get_parameter_schema(self) -> dict:
        return {
            "ma_fast": {"type": "integer", "default": 20, "description": "Fast MA period"},
            "ma_slow": {"type": "integer", "default": 50, "description": "Slow MA period"},
            "macd_fast": {"type": "integer", "default": 12, "description": "MACD fast EMA"},
            "macd_slow": {"type": "integer", "default": 26, "description": "MACD slow EMA"},
            "macd_signal": {"type": "integer", "default": 9, "description": "MACD signal period"},
            "atr_period": {"type": "integer", "default": 14, "description": "ATR period"},
            "atr_stop_mult": {"type": "number", "default": 2.5, "description": "ATR trailing stop multiplier"},
            "max_hold": {"type": "integer", "default": 40, "description": "Max holding days"}
        }