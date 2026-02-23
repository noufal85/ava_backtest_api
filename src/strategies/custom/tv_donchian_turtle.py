from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

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
class TvDonchianTurtle(BaseStrategy):
    name = "tv_donchian_turtle"
    version = "1.0.0"
    description = "Donchian Channel Turtle: Classic turtle trading system"
    category = "momentum"
    tags = ["donchian", "turtle", "breakout"]

    def __init__(self, entry_length: int = 20, exit_length: int = 10):
        self.entry_length = entry_length
        self.exit_length = exit_length

    def get_parameter_schema(self) -> dict:
        return {
            "entry_length": {"type": "integer", "default": 20, "description": "Entry channel period"},
            "exit_length": {"type": "integer", "default": 10, "description": "Exit channel period"},
        }

    def get_warmup_periods(self) -> int:
        return max(self.entry_length, self.exit_length) + 2

    def generate_signal(self, window) -> Optional[Signal]:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < max(self.entry_length, self.exit_length) + 2:
            return None

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()

        # Entry Donchian Channels
        entry_upper = max(highs[-self.entry_length-1:-1])
        entry_lower = min(lows[-self.entry_length-1:-1])

        # Exit Donchian Channels
        exit_upper = max(highs[-self.exit_length-1:-1])
        exit_lower = min(lows[-self.exit_length-1:-1])

        # Use shifted values to prevent look-ahead bias
        prev_close = closes[-2]
        prev_entry_upper = entry_upper
        prev_entry_lower = entry_lower
        prev_exit_upper = exit_upper
        prev_exit_lower = exit_lower

        # Entry signals - breakout conditions
        long_entry = (prev_close <= prev_entry_upper) and (closes[-1] > prev_entry_upper)
        short_entry = (prev_close >= prev_entry_lower) and (closes[-1] < prev_entry_lower)

        # Exit signals - opposite channel breakout
        long_exit = (prev_close >= prev_exit_lower) and (closes[-1] < prev_exit_lower)
        short_exit = (prev_close <= prev_exit_upper) and (closes[-1] > prev_exit_upper)

        if long_entry:
            return Signal(
                action="buy",
                strength=1.0,
                confidence=1.0,
                metadata={
                    "entry_upper": entry_upper,
                    "entry_lower": entry_lower,
                    "exit_upper": exit_upper,
                    "exit_lower": exit_lower
                }
            )
        elif long_exit:
            return Signal(
                action="sell",
                strength=1.0,
                confidence=1.0,
                metadata={
                    "entry_upper": entry_upper,
                    "entry_lower": entry_lower,
                    "exit_upper": exit_upper,
                    "exit_lower": exit_lower
                }
            )
        else:
            return None