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

def bollinger_bands(series: pl.Series, period: int, std_dev: float) -> dict[str, pl.Series]:
    """Calculates Bollinger Bands."""
    basis = series.rolling(window=period, center=False).mean().alias("middle")
    std = series.rolling(window=period, center=False).std().alias("std")
    upper = (basis + std_dev * std).alias("upper")
    lower = (basis - std_dev * std).alias("lower")
    return {"upper": upper, "lower": lower, "middle": basis}

@register
class TvBbWinnerLite(BaseStrategy):
    name = "tv_bb_winner_lite"
    version = "1.0.2"
    description = "Simple Bollinger Bands mean reversion with candle penetration logic"
    category = "mean_reversion"
    tags = ["mean_reversion", "bollinger_bands"]

    def __init__(self, bb_length: int = 20, bb_mult: float = 2.0, candleper: float = 30.0, show_short: bool = False):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.candleper = candleper / 100.0
        self.show_short = show_short

    def get_parameter_schema(self) -> dict:
        return {
            "bb_length": {"type": "integer", "default": 20, "description": "Bollinger Bands period"},
            "bb_mult": {"type": "number", "default": 2.0, "description": "Bollinger Bands multiplier"},
            "candleper": {"type": "number", "default": 30.0, "description": "Candle % for entry zone calculation"},
            "show_short": {"type": "boolean", "default": False, "description": "Enable short entries"},
        }

    def get_warmup_periods(self) -> int:
        return self.bb_length + 5

    def generate_signal(self, window) -> Optional[Signal]:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.bb_length:
            return None

        # Bollinger Bands Calculation
        close_series = df["close"]
        bb_data = bollinger_bands(close_series, self.bb_length, self.bb_mult)
        bb_upper = bb_data["upper"].to_list()
        bb_lower = bb_data["lower"].to_list()
        bb_basis = bb_data["middle"].to_list()

        # Candle zones
        candle_size = (df["high"] - df["low"]).to_list()
        buyzone = [(candle_size[i] * self.candleper) + df["low"][i] for i in range(len(df))]
        sellzone = [df["high"][i] - (candle_size[i] * self.candleper) for i in range(len(df))]

        # Use shifted indicators to avoid look-ahead bias
        if len(buyzone) < 2:
            return None

        prev_buyzone = buyzone[-2]
        prev_sellzone = sellzone[-2]
        prev_bb_lower = bb_lower[-2]
        prev_bb_upper = bb_upper[-2]

        # Entry signals
        long_signal = prev_buyzone < prev_bb_lower
        short_signal = (prev_buyzone > prev_bb_upper) and self.show_short

        # Exit signals (using current values)
        current_buyzone = buyzone[-1]
        current_sellzone = sellzone[-1]
        current_bb_upper = bb_upper[-1]
        current_bb_lower = bb_lower[-1]

        long_exit = current_buyzone > current_bb_upper
        short_exit = current_sellzone < current_bb_lower

        # Generate signals
        if long_signal:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"strategy": self.name, "buyzone": prev_buyzone, "bb_lower": prev_bb_lower})
        elif short_signal:
            return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"strategy": self.name, "sellzone": prev_sellzone, "bb_upper": prev_bb_upper})
        elif long_exit:
            return Signal(action="sell", strength=0.9, confidence=0.7, metadata={"reason": "bb_upper_touch"})
        elif short_exit:
            return Signal(action="buy", strength=0.9, confidence=0.7, metadata={"reason": "bb_lower_touch"})
        else:
            return None