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

def ema(series: list[float], period: int) -> list[float]:
    """Calculates the Exponential Moving Average (EMA) of a series."""
    ema = [None] * len(series)
    multiplier = 2 / (period + 1)
    # Initialize EMA with the first data point
    ema[period-1] = sum(series[:period]) / period if len(series) >= period else None

    # Calculate EMA for the rest of the series
    for i in range(period, len(series)):
        ema[i] = (series[i] - ema[i-1]) * multiplier + ema[i-1] if ema[i-1] is not None else None
    return ema

def atr(high: list[float], low: list[float], close: list[float], period: int) -> list[float]:
    """Calculates the Average True Range (ATR) of a series."""
    tr = [0.0] * len(high)
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_values = [None] * len(high)
    atr_values[period-1] = sum(tr[1:period+1]) / period if len(high) >= period else None
    for i in range(period, len(high)):
        atr_values[i] = (atr_values[i-1] * (period - 1) + tr[i]) / period if atr_values[i-1] is not None else None
    return atr_values

def keltner_channel(close: list[float], high: list[float], low: list[float], ema_period: int, atr_period: int, atr_mult: float) -> tuple[list[float], list[float], list[float]]:
    """Calculates the Keltner Channel."""
    middle = ema(close, ema_period)
    channel_width = atr(high, low, close, atr_period)
    upper = [middle[i] + atr_mult * channel_width[i] if middle[i] is not None and channel_width[i] is not None else None for i in range(len(close))]
    lower = [middle[i] - atr_mult * channel_width[i] if middle[i] is not None and channel_width[i] is not None else None for i in range(len(close))]
    return upper, middle, lower

@register
class KeltnerReversionFast(BaseStrategy):
    name = "keltner_reversion_fast"
    version = "1.0.0"
    description = "Mean reversion: buy below lower / short above upper Keltner channel"
    category = "mean_reversion"
    tags = ["mean_reversion", "keltner_channel"]

    def __init__(self, ema_period: int = 12, atr_period: int = 10, atr_mult: float = 1.8, max_hold: int = 8):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.max_hold = max_hold

    def get_warmup_periods(self) -> int:
        return max(self.ema_period, self.atr_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        close = df["close"].to_list()
        high = df["high"].to_list()
        low = df["low"].to_list()

        if len(close) < max(self.ema_period, self.atr_period):
            return None

        upper, middle, lower = keltner_channel(close, high, low, self.ema_period, self.atr_period, self.atr_mult)

        prev_close = close[-2] if len(close) > 1 else None
        prev_upper = upper[-2] if len(upper) > 1 else None
        prev_lower = lower[-2] if len(lower) > 1 else None

        if prev_close is None or prev_upper is None or prev_lower is None:
            return None

        if prev_close < prev_lower:
            strength = prev_lower - prev_close
            return Signal(action="buy", strength=strength, confidence=1.0, metadata={"prev_close": prev_close, "prev_lower": prev_lower})
        elif prev_close > prev_upper:
            strength = prev_close - prev_upper
            return Signal(action="sell", strength=strength, confidence=1.0, metadata={"prev_close": prev_close, "prev_upper": prev_upper})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "ema_period": {
                "title": "Ema Period",
                "type": "integer",
                "default": 12,
                "minimum": 2,
                "maximum": 100,
            },
            "atr_period": {
                "title": "Atr Period",
                "type": "integer",
                "default": 10,
                "minimum": 2,
                "maximum": 100,
            },
            "atr_mult": {
                "title": "Atr Mult",
                "type": "number",
                "default": 1.8,
                "minimum": 0.1,
                "maximum": 5.0,
                "multipleOf": 0.1,
            },
            "max_hold": {
                "title": "Max Hold",
                "type": "integer",
                "default": 8,
                "minimum": 1,
                "maximum": 50,
            },
        }