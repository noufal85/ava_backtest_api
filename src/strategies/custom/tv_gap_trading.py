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
class TvGapTrading(BaseStrategy):
    name = "tv_gap_trading"
    version = "1.0.0"
    description = "Gap Trading: Long-only gap fill strategy"
    category = "momentum"
    tags = ["gap", "mean_reversion", "long_only"]

    def __init__(self, min_gap_pct: float = 1.0, stop_loss_pct: float = 0.05, max_hold_days: int = 5):
        self.min_gap_pct = min_gap_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_days = max_hold_days

    def get_warmup_periods(self) -> int:
        return 2  # Need at least 1 period for previous close, plus a buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()

        if historical_data.is_empty() or current_bar.is_empty():
            return None

        # Calculate previous day's close
        prev_close = historical_data.select(pl.col("close").last()).item()

        # Get current bar data
        current_open = current_bar["open"].item()
        current_close = current_bar["close"].item()

        # Calculate gap down threshold
        gap_down_threshold = prev_close * (1 - self.min_gap_pct / 100)

        # Check for gap down
        gap_down = (prev_close > 0) and (current_open < gap_down_threshold)

        if gap_down:
            return Signal(
                action="buy",
                strength=1.0,
                confidence=0.8,
                metadata={
                    "gap_pct": ((current_open - prev_close) / prev_close) * 100,
                    "prev_close": prev_close,
                    "take_profit": prev_close,
                    "stop_loss": current_open * (1 - self.stop_loss_pct) if self.stop_loss_pct else None
                }
            )

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "min_gap_pct": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Minimum gap percentage to trigger trade"
                },
                "stop_loss_pct": {
                    "type": "number",
                    "default": 0.05,
                    "description": "Optional stop loss percentage below entry"
                },
                "max_hold_days": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum days to hold position"
                }
            },
            "required": ["min_gap_pct", "stop_loss_pct", "max_hold_days"]
        }