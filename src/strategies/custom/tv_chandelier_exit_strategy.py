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

@register
class TvChandelierExitStrategy(BaseStrategy):
    name = "tv_chandelier_exit_strategy"
    version = "1.0.0"
    description = "Chandelier Exit Strategy based on TradingView's implementation."
    category = "trend"
    tags = ["chandelier_exit", "trend_following"]

    def __init__(self, atr_period: int = 14, atr_multiplier: float = 3.0):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def get_warmup_periods(self) -> int:
        return self.atr_period + 10  # ATR period + buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.atr_period:
            return None

        high = df["high"].to_list()
        low = df["low"].to_list()
        close = df["close"].to_list()

        highest_high = max(high[-self.atr_period:])
        lowest_low = min(low[-self.atr_period:])

        atr = self._calculate_atr(high, low, close, self.atr_period)
        if atr is None:
            return None

        long_stop = highest_high - self.atr_multiplier * atr[-1]
        short_stop = lowest_low + self.atr_multiplier * atr[-1]

        if close[-1] > long_stop:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"long_stop": long_stop})
        elif close[-1] < short_stop:
            return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"short_stop": short_stop})
        else:
            return None

    def _calculate_atr(self, high: list[float], low: list[float], close: list[float], period: int) -> list[float] | None:
        tr_values = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            tr_values.append(tr)

        if len(tr_values) < period:
            return None

        atr = []
        atr.append(sum(tr_values[:period]) / period)

        for i in range(period, len(tr_values)):
            atr_value = (atr[-1] * (period - 1) + tr_values[i]) / period
            atr.append(atr_value)
        return atr

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "atr_period": {
                    "type": "integer",
                    "default": 14,
                    "description": "Period for calculating Average True Range (ATR).",
                },
                "atr_multiplier": {
                    "type": "number",
                    "default": 3.0,
                    "description": "Multiplier for the ATR value in Chandelier Exit calculation.",
                },
            },
        }