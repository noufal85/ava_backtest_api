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
class InsideBarBreakout(BaseStrategy):
    name = "inside_bar_breakout"
    version = "1.0.0"
    description = "Breakout: buy when price breaks above inside bar high"
    category = "momentum"
    tags = ["breakout", "inside bar"]

    def __init__(self, max_hold: int = 5):
        self.max_hold = max_hold

    def get_parameter_schema(self) -> dict:
        return {
            "max_hold": {"type": "integer", "default": 5, "description": "Maximum holding days before forced exit"}
        }

    def get_warmup_periods(self) -> int:
        return 2  # Need at least 2 bars to determine inside bar

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()

        combined_data = pl.concat([historical_data, current_bar])

        if len(combined_data) < 2:
            return None

        high = combined_data["high"].to_list()
        low = combined_data["low"].to_list()
        open_price = combined_data["open"].to_list()

        # Inside bar logic
        def is_inside_bar(high, low):
            if len(high) < 2 or len(low) < 2:
                return False
            return high[-1] < high[-2] and low[-1] > low[-2]

        if is_inside_bar(high[:-1], low[:-1]):  # Check if previous bar was an inside bar
            if open_price[-1] > high[-2]:
                return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"inside_bar_high": high[-2], "inside_bar_low": low[-2]})
            elif open_price[-1] < low[-2]:
                return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"inside_bar_high": high[-2], "inside_bar_low": low[-2]})

        return None