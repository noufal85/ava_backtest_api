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

def realized_vol(data: pl.DataFrame, period: int = 10) -> pl.Series:
    """Compute realized volatility from high, low, close prices."""
    log_returns = (pl.Series(np.log(data["close"]))).diff().drop_nulls()
    return log_returns.rolling_mean(period).std() * math.sqrt(252)

@register
class VixMeanReversion(BaseStrategy):
    name = "vix_mean_reversion"
    version = "1.0.0"
    description = "Vol mean reversion: buy on short-term vol spike, hold N days"
    category = "mean_reversion"
    tags = ["mean_reversion", "volatility"]

    def __init__(self, short_vol_period: int = 10, long_vol_period: int = 60, vol_spike_mult: float = 1.5, vol_complacency_mult: float = 0.7, hold_days: int = 5):
        self.short_vol_period = short_vol_period
        self.long_vol_period = long_vol_period
        self.vol_spike_mult = vol_spike_mult
        self.vol_complacency_mult = vol_complacency_mult
        self.hold_days = hold_days
        self.hold_count = 0
        self.in_trade = False
        self.entry_price = 0.0
        self.entry_date = None
        self.direction = None

    def get_parameter_schema(self) -> dict:
        return {
            "short_vol_period": {"type": "integer", "default": 10, "description": "Short Vol Period"},
            "long_vol_period": {"type": "integer", "default": 60, "description": "Long Vol Period"},
            "vol_spike_mult": {"type": "number", "default": 1.5, "description": "Vol Spike Mult"},
            "vol_complacency_mult": {"type": "number", "default": 0.7, "description": "Vol Complacency Mult"},
            "hold_days": {"type": "integer", "default": 5, "description": "Hold Days"},
        }

    def get_warmup_periods(self) -> int:
        return self.long_vol_period + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        combined_data = pl.concat([historical_data, current_bar])

        if len(combined_data) < self.long_vol_period:
            return None

        short_vol = realized_vol(combined_data, period=self.short_vol_period).to_list()[-1]
        long_vol = realized_vol(combined_data, period=self.long_vol_period).to_list()[-1]

        if long_vol == 0:
            return None

        vol_ratio = short_vol / long_vol

        if not self.in_trade:
            if vol_ratio > self.vol_spike_mult:
                self.in_trade = True
                self.hold_count = 0
                self.entry_price = current_bar["close"][0]
                self.entry_date = current_bar["timestamp"][0]
                self.direction = "buy"
                strength = min(1.0, vol_ratio - self.vol_spike_mult)
                return Signal(action="buy", strength=strength, confidence=1.0, metadata={"vol_ratio": vol_ratio})
            elif vol_ratio < self.vol_complacency_mult:
                self.in_trade = True
                self.hold_count = 0
                self.entry_price = current_bar["close"][0]
                self.entry_date = current_bar["timestamp"][0]
                self.direction = "sell"
                strength = min(1.0, self.vol_complacency_mult - vol_ratio)
                return Signal(action="sell", strength=strength, confidence=1.0, metadata={"vol_ratio": vol_ratio})
            else:
                return None
        else:
            self.hold_count += 1
            if self.hold_count > self.hold_days:
                self.in_trade = False
                self.entry_price = 0.0
                self.entry_date = None
                self.direction = None
                return None
            else:
                return None