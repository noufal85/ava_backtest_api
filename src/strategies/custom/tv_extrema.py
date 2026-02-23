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
class TvExtrema(BaseStrategy):
    name = "tv_extrema"
    version = "1.0.0"
    description = "Mathematical extrema detection with exponential timeframes (simplified)"
    category = "multi_factor"
    tags = ["extrema", "multi-timeframe"]

    def __init__(self, base: float = 2.0, factor: float = 1.0, crop: int = 3, max_periods: int = 6):
        self.base = base
        self.factor = factor
        self.crop = crop
        self.max_periods = max_periods

    def get_warmup_periods(self) -> int:
        periods = []
        for i in range(self.max_periods):
            period = int(round(pow(self.base, self.factor * (i + self.crop))))
            periods.append(period)
        return max(periods, default=0) + 5  # Add a small buffer

    def generate_signal(self, window) -> Signal | None:
        df = pl.concat([window.historical(), window.current_bar()])
        
        periods = []
        for i in range(self.max_periods):
            period = int(round(pow(self.base, self.factor * (i + self.crop))))
            if period <= len(df):
                periods.append(period)

        low_values = []
        high_values = []
        for period in periods:
            low_values.append(df["low"].rolling(period, min_periods=1).min().to_list())
            high_values.append(df["high"].rolling(period, min_periods=1).max().to_list())

        fall_low = [False] * len(df)
        rise_high = [False] * len(df)

        for i in range(len(periods)):
            current_fall_low = [False] * len(df)
            current_rise_high = [False] * len(df)
            
            for j in range(1, len(df)):
                if low_values[i][j] < low_values[i][j-1]:
                    current_fall_low[j] = True
                if high_values[i][j] > high_values[i][j-1]:
                    current_rise_high[j] = True
            
            for j in range(len(df)):
                if current_fall_low[j]:
                    fall_low[j] = True
                if current_rise_high[j]:
                    rise_high[j] = True

        kick_low = False
        kick_high = False
        if len(df) > 1:
            kick_low = fall_low[-2] and not fall_low[-1]
            kick_high = rise_high[-2] and not rise_high[-1]

        if kick_low:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"signal_type": "kick_low"})
        elif kick_high:
            return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"signal_type": "kick_high"})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "base": {"type": "number", "default": 2.0},
                "factor": {"type": "number", "default": 1.0},
                "crop": {"type": "integer", "default": 3},
                "max_periods": {"type": "integer", "default": 6},
            },
            "required": ["base", "factor", "crop", "max_periods"],
        }