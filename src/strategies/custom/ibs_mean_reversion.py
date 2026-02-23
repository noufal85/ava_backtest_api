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
class IbsMeanReversion(BaseStrategy):
    name = "ibs_mean_reversion"
    version = "1.0.0"
    description = "IBS mean reversion: buy low IBS below lower band, exit on high IBS"
    category = "mean_reversion"
    tags = ["mean_reversion", "ibs"]

    def __init__(self, 
                 range_period: int = 25,
                 high_period: int = 10,
                 band_multiplier: float = 2.5,
                 ibs_entry_threshold: float = 0.3,
                 ibs_exit_threshold: float = 0.8,
                 stop_multiplier: float = 3.0):
        self.range_period = range_period
        self.high_period = high_period
        self.band_multiplier = band_multiplier
        self.ibs_entry_threshold = ibs_entry_threshold
        self.ibs_exit_threshold = ibs_exit_threshold
        self.stop_multiplier = stop_multiplier

    def get_warmup_periods(self) -> int:
        return max(self.range_period, self.high_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < max(self.range_period, self.high_period):
            return None

        # Average true range (high - low) over range_period
        bar_range = df["high"] - df["low"]
        avg_range = bar_range.rolling(window=self.range_period, min_periods=self.range_period).mean()

        # IBS: Internal Bar Strength
        ibs = (df["close"] - df["low"]) / bar_range.fill_nan(0.000001)

        # Rolling high over high_period
        rolling_high = df["high"].rolling(window=self.high_period, min_periods=self.high_period).max()

        # Lower band
        lower_band = rolling_high - self.band_multiplier * avg_range

        # Use previous bar values to avoid look-ahead bias
        prev_close = df["close"].shift(1).to_numpy()[-1]
        prev_ibs = ibs.shift(1).to_numpy()[-1]
        prev_lower_band = lower_band.shift(1).to_numpy()[-1]

        if (prev_close < prev_lower_band) and (prev_ibs < self.ibs_entry_threshold):
            signal_strength = self.ibs_entry_threshold - prev_ibs
            return Signal(
                action="buy",
                strength=float(min(1.0, signal_strength)),
                confidence=1.0,
                metadata={"ibs": float(prev_ibs)}
            )

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "range_period": {
                    "type": "integer",
                    "default": 25,
                    "description": "Lookback for average range"
                },
                "high_period": {
                    "type": "integer",
                    "default": 10,
                    "description": "Lookback for rolling high"
                },
                "band_multiplier": {
                    "type": "number",
                    "default": 2.5,
                    "description": "Multiplier for avg_range in lower band"
                },
                "ibs_entry_threshold": {
                    "type": "number",
                    "default": 0.3,
                    "description": "IBS must be below this to enter"
                },
                "ibs_exit_threshold": {
                    "type": "number",
                    "default": 0.8,
                    "description": "IBS above this triggers exit"
                },
                "stop_multiplier": {
                    "type": "number",
                    "default": 3.0,
                    "description": "Stop loss = entry - stop_multiplier * avg_range"
                }
            },
            "required": ["range_period", "high_period", "band_multiplier", "ibs_entry_threshold", "ibs_exit_threshold", "stop_multiplier"]
        }