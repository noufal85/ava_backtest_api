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
class StopLossHuntReversal(BaseStrategy):
    name = "stop_loss_hunt_reversal"
    version = "1.0.0"
    description = "Stop-loss hunt reversal: buy reclaims of broken support with high volume"
    category = "multi_factor"
    tags = ["reversal", "volume", "support"]

    def __init__(self, support_period: int = 20, volume_multiplier: float = 1.5, volume_avg_period: int = 20, reward_ratio: float = 2.0, max_hold_days: int = 10):
        self.support_period = support_period
        self.volume_multiplier = volume_multiplier
        self.volume_avg_period = volume_avg_period
        self.reward_ratio = reward_ratio
        self.max_hold_days = max_hold_days

    def get_warmup_periods(self) -> int:
        return max(self.support_period, self.volume_avg_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < max(self.support_period, self.volume_avg_period):
            return None

        # Support level: lowest low over support_period (excluding current bar)
        lows = df["low"].to_list()
        support_levels = []
        for i in range(len(lows)):
            if i < self.support_period:
                support_levels.append(None)
            else:
                support_levels.append(min(lows[i-self.support_period:i]))

        # Volume average
        volumes = df["volume"].to_list()
        vol_avgs = []
        for i in range(len(volumes)):
            if i < self.volume_avg_period:
                vol_avgs.append(None)
            else:
                vol_avgs.append(sum(volumes[i-self.volume_avg_period:i]) / self.volume_avg_period)

        # Hunt candle: previous bar dipped below support with high volume
        if len(df) < 2:
            return None

        prev_low = lows[-2]
        prev_close = df["close"][-2]
        prev_volume = volumes[-2]
        prev_support = support_levels[-2]
        prev_vol_avg = vol_avgs[-2]

        # Current bar (reclaim): closes above the support level that was broken
        curr_close = df["close"][-1]

        if prev_support is None or prev_vol_avg is None:
            return None

        hunt_occurred = (prev_low < prev_support) and (prev_volume >= self.volume_multiplier * prev_vol_avg)
        reclaim = curr_close > prev_support

        if hunt_occurred and reclaim:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"hunt_low": prev_low, "hunt_support": prev_support})

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "support_period": {
                    "type": "integer",
                    "default": 20,
                    "description": "Lookback for support level"
                },
                "volume_multiplier": {
                    "type": "number",
                    "default": 1.5,
                    "description": "Required volume vs avg"
                },
                "volume_avg_period": {
                    "type": "integer",
                    "default": 20,
                    "description": "Volume average lookback"
                },
                "reward_ratio": {
                    "type": "number",
                    "default": 2.0,
                    "description": "Profit target as multiple of risk"
                },
                "max_hold_days": {
                    "type": "integer",
                    "default": 10,
                    "description": "Time stop in days"
                }
            },
            "required": ["support_period", "volume_multiplier", "volume_avg_period", "reward_ratio", "max_hold_days"]
        }