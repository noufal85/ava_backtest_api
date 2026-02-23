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

def atr(df: pl.DataFrame, period: int) -> pl.Series:
    """Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1).fill_null(0.0))
    tr3 = abs(low - close.shift(1).fill_null(0.0))
    tr = pl.max([tr1, tr2, tr3])

    atr_values = []
    for i in range(len(df)):
        if i < period:
            atr_values.append(None)
        else:
            atr_values.append(tr[i-period+1:i+1].mean())
    return pl.Series(atr_values)

@register
class TurtleChannel(BaseStrategy):
    name = "turtle_channel"
    version = "1.1.0"
    description = "Turtle Channel: Donchian breakout/breakdown entry with exits (long & short)"
    category = "momentum"
    tags = ["trend", "breakout"]

    def __init__(self, entry_period: int = 20, exit_period: int = 10, atr_period: int = 14, min_price: float = 5.0):
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.min_price = min_price

    def get_parameter_schema(self) -> dict:
        return {
            "entry_period": {"type": "integer", "default": 20, "description": "Donchian channel period for entry"},
            "exit_period": {"type": "integer", "default": 10, "description": "Donchian channel period for exit"},
            "atr_period": {"type": "integer", "default": 14, "description": "ATR period"},
            "min_price": {"type": "number", "default": 5.0, "description": "Minimum price to consider"}
        }

    def get_warmup_periods(self) -> int:
        return max(self.entry_period, self.exit_period, self.atr_period) + 5

    def generate_signal(self, window) -> Signal | None:
        df = window.historical()
        if df is None or len(df) < max(self.entry_period, self.exit_period, self.atr_period):
            return None

        current_bar = window.current_bar()
        if current_bar is None:
            return None

        df = pl.concat([df, current_bar])
        
        # Donchian Channels
        high_prices = df["high"].to_list()
        low_prices = df["low"].to_list()
        close_prices = df["close"].to_list()

        if len(high_prices) < self.entry_period or len(low_prices) < self.entry_period:
            return None

        donchian_high_entry = max(high_prices[-self.entry_period:-1]) if len(high_prices) >= self.entry_period else None
        donchian_low_entry = min(low_prices[-self.entry_period:-1]) if len(low_prices) >= self.entry_period else None
        donchian_high_exit = max(high_prices[-self.exit_period:-1]) if len(high_prices) >= self.exit_period else None
        donchian_low_exit = min(low_prices[-self.exit_period:-1]) if len(low_prices) >= self.exit_period else None
        donchian_mid_exit = (donchian_high_exit + donchian_low_exit) / 2.0 if donchian_high_exit is not None and donchian_low_exit is not None else None

        current_close = close_prices[-1]

        if current_close < self.min_price:
            return None

        if donchian_high_entry is not None and current_close > donchian_high_entry:
            strength = (current_close - donchian_high_entry) / donchian_high_entry
            return Signal(action="buy", strength=strength, confidence=1.0)
        elif donchian_low_entry is not None and current_close < donchian_low_entry:
            strength = (donchian_low_entry - current_close) / donchian_low_entry
            return Signal(action="sell", strength=strength, confidence=1.0)
        elif donchian_low_exit is not None and current_close < donchian_low_exit:
            return Signal(action="sell", strength=0.0, confidence=1.0)
        elif donchian_mid_exit is not None and current_close > donchian_mid_exit:
            return Signal(action="buy", strength=0.0, confidence=1.0)
        else:
            return None