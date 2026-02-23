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

@register
class GapAndGo(BaseStrategy):
    name = "gap_and_go"
    version = "1.0.0"
    description = "Gap and Go: buy gap-up stocks for intraday momentum continuation"
    category = "momentum"
    tags = ["gap", "momentum"]

    def __init__(self, min_gap_pct: float = 0.02, max_gap_pct: float = 0.08, stop_pct: float = 0.01):
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.stop_pct = stop_pct

    def get_parameter_schema(self) -> dict:
        return {
            "min_gap_pct": {"type": "number", "default": 0.02, "description": "Minimum gap percentage"},
            "max_gap_pct": {"type": "number", "default": 0.08, "description": "Maximum gap percentage"},
            "stop_pct": {"type": "number", "default": 0.01, "description": "Stop loss percentage"},
        }

    def get_warmup_periods(self) -> int:
        return 2  # Need at least 1 period to calculate previous close + small buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()

        if historical_data.is_empty() or current_bar.is_empty():
            return None

        # Calculate previous close
        if historical_data.height < 1:
            return None

        prev_close = historical_data["close"][-1]

        open_price = current_bar["open"].item()
        close_price = current_bar["close"].item()
        high_price = current_bar["high"].item()
        low_price = current_bar["low"].item()

        if prev_close is None or math.isnan(prev_close) or prev_close == 0:
            return None

        gap_pct = (open_price - prev_close) / prev_close

        if self.min_gap_pct <= gap_pct <= self.max_gap_pct:
            return Signal(
                action="buy",
                strength=float(gap_pct),
                confidence=1.0,
                metadata={"gap_pct": float(gap_pct)},
            )
        elif -self.min_gap_pct >= gap_pct >= -self.max_gap_pct:
            return Signal(
                action="sell",
                strength=float(abs(gap_pct)),
                confidence=1.0,
                metadata={"gap_pct": float(gap_pct)},
            )
        else:
            return None