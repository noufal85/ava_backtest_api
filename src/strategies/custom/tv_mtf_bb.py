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
class TvMtfBb(BaseStrategy):
    name = "tv_mtf_bb"
    version = "1.0.0"
    description = "MtfBb Strategy: STUB - Multi-timeframe strategy requiring higher timeframe data not available"
    category = "multi_factor"
    tags = []

    def __init__(self):
        pass

    def get_warmup_periods(self) -> int:
        return 1  # Minimal warmup since the strategy is a stub

    def generate_signal(self, window) -> Signal | None:
        # This is a stub strategy, so it doesn't generate any signals.
        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {},
        }