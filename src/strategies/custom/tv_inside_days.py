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

def rsi(prices: list[float], period: int = 14) -> float:
    """Calculates the Relative Strength Index (RSI) for a given list of prices."""
    if len(prices) < period + 1:
        return np.nan

    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0.0
    rsi = 100. - 100. / (1. + rs)

    for i in range(period+1, len(deltas)):
        delta = deltas[i]
        if delta >= 0:
            up = (up * (period - 1) + delta) / period
            down = (down * (period - 1)) / period
        else:
            up = (up * (period - 1)) / period
            down = (down * (period - 1) - delta) / period

        rs = up / down if down != 0 else 0.0
        rsi = 100. - 100. / (1. + rs)

    return float(rsi)

def is_inside_bar(high: list[float], low: list[float]) -> bool:
    """Checks if the current bar is an inside bar."""
    if len(high) < 2 or len(low) < 2:
        return False

    return high[-1] < high[-2] and low[-1] > low[-2]

@register
class TvInsideDays(BaseStrategy):
    name = "tv_inside_days"
    version = "1.0.0"
    description = "Inside Day Strategy: Enters on inside day patterns with RSI exit"
    category = "momentum"
    tags = ["inside_day", "rsi"]

    def __init__(self, rsi_length: int = 5, overbought_level: float = 80.0):
        self.rsi_length = rsi_length
        self.overbought_level = overbought_level

    def get_warmup_periods(self) -> int:
        return self.rsi_length + 2 # Add a small buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()

        combined_data = pl.concat([historical_data, current_bar])

        high_prices = combined_data["high"].to_list()
        low_prices = combined_data["low"].to_list()
        close_prices = combined_data["close"].to_list()

        if len(high_prices) < 2 or len(low_prices) < 2:
            return None

        inside_day = is_inside_bar(high_prices, low_prices)
        rsi_value = rsi(close_prices, self.rsi_length)

        if not np.isnan(rsi_value):
            if len(historical_data) > 0:
                previous_highs = historical_data["high"].to_list()
                previous_lows = historical_data["low"].to_list()

                if len(previous_highs) > 0 and len(previous_lows) > 0:
                    if is_inside_bar(previous_highs[-2:], previous_lows[-2:]):
                        return Signal(
                            action="buy",
                            strength=1.0,
                            confidence=0.8,
                            metadata={"pattern": "inside_day"}
                        )

            if rsi_value > self.overbought_level:
                return Signal(
                    action="sell",
                    strength=1.0,
                    confidence=0.8,
                    metadata={"reason": "rsi_overbought"}
                )

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "rsi_length": {
                "type": "integer",
                "default": 5,
                "description": "RSI calculation period"
            },
            "overbought_level": {
                "type": "number",
                "default": 80.0,
                "description": "Overbought RSI threshold for exit"
            }
        }