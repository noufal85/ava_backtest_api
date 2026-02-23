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
class TvDmiWinner(BaseStrategy):
    name = "tv_dmi_winner"
    version = "1.0.0"
    description = "DmiWinner Strategy: STUB - Strategy requiring specific implementation"
    category = "multi_factor"
    tags = []

    def __init__(self):
        pass

    def get_warmup_periods(self) -> int:
        return 20  # Example: Adjust based on indicator periods

    def generate_signal(self, window) -> Signal | None:
        # Placeholder implementation - replace with actual DMI logic
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.get_warmup_periods():
            return None

        # Example: Simple moving average crossover
        closes = df["close"].to_list()
        if len(closes) < 2:
            return None

        if closes[-1] > np.mean(closes[-10:-1]):
            return Signal(action="buy", strength=0.8, confidence=0.7)
        elif closes[-1] < np.mean(closes[-10:-1]):
            return Signal(action="sell", strength=0.8, confidence=0.7)
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }