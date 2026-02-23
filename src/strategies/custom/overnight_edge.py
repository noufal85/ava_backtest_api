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

def _rsi(closes: list[float], period: int) -> list[float]:
    """Calculates the Relative Strength Index (RSI) for a given period."""
    if len(closes) < period:
        return [np.nan] * len(closes)

    rsi_values = [np.nan] * len(closes)
    for i in range(period, len(closes)):
        period_closes = closes[i-period:i]
        deltas = [period_closes[j] - period_closes[j-1] for j in range(1, len(period_closes))]
        
        avg_gain = sum([d for d in deltas if d > 0]) / period
        avg_loss = abs(sum([d for d in deltas if d < 0]) / period)

        if avg_loss == 0:
            rsi = 100.0 if avg_gain > 0 else 0.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi_values[i] = rsi
    
    return rsi_values

@register
class OvernightEdge(BaseStrategy):
    name = "overnight_edge"
    version = "1.0.0"
    description = "Overnight edge: buy at close, sell at next open, RSI filter"
    category = "multi_factor"
    tags = ["overnight", "mean_reversion", "rsi"]

    def __init__(self, rsi_period: int = 14, rsi_filter: float = 50.0, rsi_short_filter: float = 70.0):
        self.rsi_period = rsi_period
        self.rsi_filter = rsi_filter
        self.rsi_short_filter = rsi_short_filter

    def get_parameter_schema(self) -> dict:
        return {
            "rsi_period": {"type": "integer", "default": 14, "minimum": 2, "maximum": 200},
            "rsi_filter": {"type": "number", "default": 50.0, "minimum": 0.0, "maximum": 100.0},
            "rsi_short_filter": {"type": "number", "default": 70.0, "minimum": 0.0, "maximum": 100.0},
        }

    def get_warmup_periods(self) -> int:
        return self.rsi_period + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()
        rsi_values = _rsi(closes, self.rsi_period)
        current_rsi = rsi_values[-1]
        previous_rsi = rsi_values[-2] if len(rsi_values) > 1 else None

        if previous_rsi is None or math.isnan(previous_rsi):
            return None

        if previous_rsi < self.rsi_filter:
            strength = self.rsi_filter - previous_rsi
            return Signal(action="buy", strength=strength / self.rsi_filter, confidence=1.0, metadata={"rsi": previous_rsi})
        elif previous_rsi > self.rsi_short_filter:
            strength = previous_rsi - self.rsi_short_filter
            return Signal(action="sell", strength=strength / (100 - self.rsi_short_filter), confidence=1.0, metadata={"rsi": previous_rsi})
        else:
            return None