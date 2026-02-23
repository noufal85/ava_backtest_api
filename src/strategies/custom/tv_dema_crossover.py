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

def _ema(series: list[float], period: int) -> list[float]:
    """Exponential Moving Average."""
    ema = [series[0]]
    alpha = 2 / (period + 1)
    for i in range(1, len(series)):
        ema.append(alpha * series[i] + (1 - alpha) * ema[-1])
    return ema

def _dema(series: list[float], period: int) -> list[float]:
    """Double Exponential Moving Average."""
    ema1 = _ema(series, period)
    ema2 = _ema(ema1, period)
    return [2 * e1 - e2 for e1, e2 in zip(ema1, ema2)]

@register
class TvDemaCrossover(BaseStrategy):
    name = "tv_dema_crossover"
    version = "1.0.0"
    description = "DEMA Crossover Strategy based on TradingView Pine Script"
    category = "trend"
    tags = ["dema", "crossover", "tradingview"]

    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def get_warmup_periods(self) -> int:
        return max(self.fast_period, self.slow_period) + 20

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()

        if len(closes) < max(self.fast_period, self.slow_period):
            return None

        dema_fast = _dema(closes, self.fast_period)
        dema_slow = _dema(closes, self.slow_period)

        if len(dema_fast) < 2 or len(dema_slow) < 2:
            return None

        current_fast = dema_fast[-1]
        previous_fast = dema_fast[-2]
        current_slow = dema_slow[-1]
        previous_slow = dema_slow[-2]

        if previous_fast <= previous_slow and current_fast > current_slow:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"reason": "dema_crossover"})
        elif previous_fast >= previous_slow and current_fast < current_slow:
            return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"reason": "dema_crossunder"})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "fast_period": {
                    "type": "integer",
                    "default": 12,
                    "minimum": 2,
                    "description": "Period for the faster DEMA"
                },
                "slow_period": {
                    "type": "integer",
                    "default": 26,
                    "minimum": 2,
                    "description": "Period for the slower DEMA"
                },
            },
            "required": ["fast_period", "slow_period"],
        }