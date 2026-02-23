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

def atr(high: list[float], low: list[float], close: list[float], period: int) -> list[float]:
    """Average True Range (ATR) indicator."""
    true_range = []
    for i in range(len(high)):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        true_range.append(tr)

    atr_values = []
    for i in range(len(true_range)):
        if i < period:
            atr_values.append(np.nan)
        else:
            atr_values.append(np.mean(true_range[i - period + 1:i + 1]))
    return atr_values

@register
class TvAtrTrailingStop(BaseStrategy):
    name = "tv_atr_trailing_stop"
    version = "1.0.0"
    description = "ATR Trailing Stop: Dynamic trailing stop system using ATR"
    category = "volatility"
    tags = ["atr", "trailing stop", "volatility"]

    def __init__(self, atr_length: int = 14, atr_multiplier: float = 3.0):
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier

    def get_warmup_periods(self) -> int:
        return self.atr_length + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        high = df["high"].to_list()
        low = df["low"].to_list()
        close = df["close"].to_list()

        atr_values = atr(high, low, close, self.atr_length)

        if len(atr_values) < 2 or math.isnan(atr_values[-1]) or math.isnan(atr_values[-2]):
            return None

        long_stop_level = [c - a * self.atr_multiplier for c, a in zip(close, atr_values)]
        short_stop_level = [c + a * self.atr_multiplier for c, a in zip(close, atr_values)]

        trail_stop = [np.nan] * len(close)
        direction = [0] * len(close)

        if len(close) > 0:
            trail_stop[0] = long_stop_level[0]
            direction[0] = 1
        
        for i in range(1, len(close)):
            if direction[i-1] == 1:
                trail_stop[i] = max(trail_stop[i-1], long_stop_level[i])
                if close[i] < trail_stop[i]:
                    direction[i] = -1
                    trail_stop[i] = short_stop_level[i]
                else:
                    direction[i] = 1
            else:
                trail_stop[i] = min(trail_stop[i-1], short_stop_level[i])
                if close[i] > trail_stop[i]:
                    direction[i] = 1
                    trail_stop[i] = long_stop_level[i]
                else:
                    direction[i] = -1

        if len(direction) < 2:
            return None

        if direction[-1] == 1 and direction[-2] == -1:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"atr": atr_values[-1], "trail_stop": trail_stop[-1]})
        elif direction[-1] == -1 and direction[-2] == 1:
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"atr": atr_values[-1], "trail_stop": trail_stop[-1]})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "atr_length": {"type": "integer", "default": 14, "minimum": 1},
            "atr_multiplier": {"type": "number", "default": 3.0, "minimum": 0.1}
        }