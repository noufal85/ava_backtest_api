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

import calendar
from datetime import datetime as _dt

def days_until_month_end(dates) -> list[int]:
    """Computes the number of calendar days until the end of the month."""
    result = []
    for d in dates:
        if hasattr(d, 'date'):
            d = d.date()  # convert datetime to date
        last_day_num = calendar.monthrange(d.year, d.month)[1]
        result.append(last_day_num - d.day)
    return result

def trading_day_of_month(dates) -> list[int]:
    """Computes the trading day of the month."""
    result = []
    for d in dates:
        if hasattr(d, 'date'):
            d = d.date()
        result.append(d.day)
    return result

@register
class TurnOfMonthStrategy(BaseStrategy):
    name = "turn_of_month"
    version = "1.0.0"
    description = "Calendar: buy near month end, sell early in new month"
    category = "multi_factor"
    tags = ["calendar", "mean_reversion"]

    def __init__(self, entry_days_before_end: int = 3, exit_days_after_start: int = 3):
        self.entry_days_before_end = entry_days_before_end
        self.exit_days_after_start = exit_days_after_start

    def get_warmup_periods(self) -> int:
        return max(self.entry_days_before_end, self.exit_days_after_start) + 5

    def generate_signal(self, window) -> Signal | None:
        current_bar = window.current_bar()
        historical_data = window.historical()

        all_data = pl.concat([historical_data, current_bar])
        dates = all_data["ts"].to_list()

        days_to_end = days_until_month_end(dates)
        day_of_month = trading_day_of_month(dates)

        entry_days = self.entry_days_before_end
        exit_day = self.exit_days_after_start

        buy_mask = days_to_end[-1] == entry_days
        sell_mask = day_of_month[-1] == exit_day

        if buy_mask:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"reason": "turn_of_month_entry"})
        elif sell_mask:
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"reason": "turn_of_month_exit"})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "entry_days_before_end": {
                    "type": "integer",
                    "default": 3,
                    "description": "Trading days before month end to enter",
                },
                "exit_days_after_start": {
                    "type": "integer",
                    "default": 3,
                    "description": "Trading day of new month to exit",
                },
            },
        }