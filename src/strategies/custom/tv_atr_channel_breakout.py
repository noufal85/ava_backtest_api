from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import polars as pl
import numpy as np
import math

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

from src.core.strategy.registry import register

def sma(closes: list[float], period: int) -> list[float]:
    """Simple Moving Average."""
    if len(closes) < period:
        return [None] * len(closes)
    
    sma_values = []
    for i in range(len(closes)):
        if i < period - 1:
            sma_values.append(None)
        else:
            window = closes[i - period + 1 : i + 1]
            sma_values.append(sum(window) / period)
    return sma_values

def atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float]:
    """Average True Range."""
    tr_values = []
    for i in range(len(highs)):
        if i == 0:
            tr = highs[i] - lows[i]
        else:
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_values.append(tr)

    atr_values = []
    for i in range(len(tr_values)):
        if i < period:
            atr_values.append(None)
        else:
            atr_values.append(sum(tr_values[i-period:i]) / period)
    return atr_values

@register
class TvAtrChannelBreakout(BaseStrategy):
    name = "tv_atr_channel_breakout"
    version = "1.0.0"
    description = "ATR Channel Breakout: Trades breakouts of ATR-based channels"
    category = "momentum"
    tags = ["atr", "channel", "breakout"]

    def __init__(self, length: int = 20, mult: float = 2.0):
        self.length = length
        self.mult = mult

    def get_parameter_schema(self) -> dict:
        return {
            "length": {"type": "integer", "default": 20, "minimum": 1},
            "mult": {"type": "number", "default": 2.0, "minimum": 0.1}
        }

    def get_warmup_periods(self) -> int:
        return self.length + 5

    def generate_signal(self, window) -> Signal | None:
        historical_df = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_df, current_bar])

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()

        basis = sma(closes, self.length)
        atr_values = atr(highs, lows, closes, self.length)

        if basis[-1] is None or atr_values[-1] is None:
            return None

        upper = basis[-1] + self.mult * atr_values[-1]
        lower = basis[-1] - self.mult * atr_values[-1]

        if len(closes) < 2:
            return None

        prev_close = closes[-2]
        current_close = closes[-1]

        if (prev_close <= upper) and (current_close > upper):
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"upper": upper, "lower": lower, "basis": basis[-1]})
        #if (prev_close >= lower) and (current_close < lower):
        #    return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"upper": upper, "lower": lower, "basis": basis[-1]})

        return None