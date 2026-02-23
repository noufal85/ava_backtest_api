from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

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

def sma(series: pl.Series, period: int) -> pl.Series:
    """Simple Moving Average."""
    if len(series) < period:
        return pl.Series([None] * len(series))
    
    sma_values = []
    for i in range(len(series)):
        if i < period - 1:
            sma_values.append(None)
        else:
            window = series[i - period + 1:i + 1]
            sma_values.append(window.mean())
    return pl.Series(sma_values)

def volume_sma(series: pl.Series, period: int) -> pl.Series:
    """Simple Moving Average for Volume."""
    return sma(series, period)

@register
class DualMomentum(BaseStrategy):
    name = "dual_momentum"
    version = "1.1.0"
    description = "Dual momentum: relative return ranking + absolute SMA filter (long & short)"
    category = "momentum"
    tags = ["momentum", "relative_momentum", "absolute_momentum"]

    def __init__(self, lookback_days: int = 20, sma_period: int = 20, top_n: int = 5):
        self.lookback_days = lookback_days
        self.sma_period = sma_period
        self.top_n = top_n

    def get_parameter_schema(self) -> dict:
        return {
            "lookback_days": {"type": "integer", "default": 20},
            "sma_period": {"type": "integer", "default": 20},
            "top_n": {"type": "integer", "default": 5},
        }

    def get_warmup_periods(self) -> int:
        return max(self.lookback_days, self.sma_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < max(self.lookback_days, self.sma_period):
            return None

        close_prices = df["close"]
        volumes = df["volume"]

        # Momentum calculation
        if len(close_prices) > self.lookback_days:
            momentum = (close_prices[-1] - close_prices[-self.lookback_days -1]) / close_prices[-self.lookback_days -1] * 100.0
        else:
            return None

        # SMA calculation
        sma_values = sma(close_prices, self.sma_period)
        current_sma = sma_values[-1]

        # Volume SMA calculation
        volume_sma_values = volume_sma(volumes, self.sma_period)
        current_volume_sma = volume_sma_values[-1]

        # Filters
        min_price = 5.0
        min_volume = 100_000

        current_close = close_prices[-1]
        if current_close < min_price or volumes[-1] < min_volume:
            return None

        # Generate signal
        if current_close > current_sma and momentum > 0:
            return Signal(
                action="buy",
                strength=min(momentum / 100.0, 1.0),
                confidence=0.7,
                metadata={
                    "momentum": momentum,
                    "sma": current_sma,
                    "close": current_close,
                },
            )
        elif current_close < current_sma and momentum < 0:
            return Signal(
                action="sell",
                strength=min(abs(momentum) / 100.0, 1.0),
                confidence=0.7,
                metadata={
                    "momentum": momentum,
                    "sma": current_sma,
                    "close": current_close,
                },
            )
        else:
            return None