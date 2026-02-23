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

def bollinger_bands(series: pl.Series, period: int, std_dev: float) -> dict[str, pl.Series]:
    """Calculates Bollinger Bands.

    Args:
        series (pl.Series): Input series (e.g., closing prices).
        period (int): Period for calculating moving average and standard deviation.
        std_dev (float): Number of standard deviations for band width.

    Returns:
        dict[str, pl.Series]: A dictionary containing the upper band, lower band, and middle band.
    """
    middle_band = series.rolling_mean(window_size=period, min_periods=period)
    std = series.rolling_std(window_size=period, min_periods=period)
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    return {"upper": upper_band, "lower": lower_band, "middle": middle_band}

@register
class TvMultiBb(BaseStrategy):
    name = "tv_multi_bb"
    version = "1.0.0"
    description = "Multi Bollinger Bands with averaged bands from different periods"
    category = "multi_factor"
    tags = ["bollinger bands", "mean reversion"]

    def __init__(self, bb_length_primary: int = 20, bb_length_secondary: int = 30, bb_mult: float = 2.0):
        self.bb_length_primary = bb_length_primary
        self.bb_length_secondary = bb_length_secondary
        self.bb_mult = bb_mult

    def get_parameter_schema(self) -> dict:
        return {
            "bb_length_primary": {"type": "integer", "default": 20, "description": "Primary Bollinger Bands period"},
            "bb_length_secondary": {"type": "integer", "default": 30, "description": "Secondary Bollinger Bands period"},
            "bb_mult": {"type": "number", "default": 2.0, "description": "Bollinger Bands multiplier"}
        }

    def get_warmup_periods(self) -> int:
        return max(self.bb_length_primary, self.bb_length_secondary) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.get_warmup_periods():
            return None

        # Primary Bollinger Bands
        bb_primary = bollinger_bands(df["close"], period=self.bb_length_primary, std_dev=self.bb_mult)
        bb_upper_primary = bb_primary["upper"]
        bb_lower_primary = bb_primary["lower"]
        bb_basis_primary = bb_primary["middle"]

        # Secondary Bollinger Bands (simulating MTF)
        bb_secondary = bollinger_bands(df["close"], period=self.bb_length_secondary, std_dev=self.bb_mult)
        bb_upper_secondary = bb_secondary["upper"]
        bb_lower_secondary = bb_secondary["lower"]
        bb_basis_secondary = bb_secondary["middle"]

        # Averaged bands
        bb_upper_avg = (bb_upper_primary + bb_upper_secondary) / 2
        bb_lower_avg = (bb_lower_primary + bb_lower_secondary) / 2
        bb_basis_avg = (bb_basis_primary + bb_basis_secondary) / 2

        # Use shifted indicators to avoid look-ahead bias
        prev_close = df["close"].shift(1).to_list()[-1]
        prev_bb_lower_avg = bb_lower_avg.shift(1).to_list()[-1]
        prev_bb_upper_primary = bb_upper_primary.shift(1).to_list()[-1]

        # Entry signals
        long_signal = prev_close < prev_bb_lower_avg

        # Exit signals
        long_exit = df["close"].to_list()[-1] > prev_bb_upper_primary

        if long_signal:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"bb_lower_avg": prev_bb_lower_avg, "bb_upper_primary": prev_bb_upper_primary})

        if long_exit:
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"bb_upper_primary": prev_bb_upper_primary})

        return None