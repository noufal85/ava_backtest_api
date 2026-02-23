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

def sma(series: list[float], period: int) -> list[float]:
    """Simple Moving Average."""
    if len(series) < period:
        return [None] * len(series)
    
    result = []
    for i in range(len(series)):
        if i < period - 1:
            result.append(None)
        else:
            window = series[i - period + 1 : i + 1]
            result.append(sum(window) / period)
    return result

def bollinger_bands(series: list[float], period: int, std_mult: float) -> tuple[list[float], list[float], list[float]]:
    """Bollinger Bands."""
    middle = sma(series, period)
    upper = [None] * len(series)
    lower = [None] * len(series)
    
    for i in range(len(series)):
        if middle[i] is not None:
            window = series[i - period + 1 : i + 1]
            std_dev = np.std(window)
            upper[i] = middle[i] + std_mult * std_dev
            lower[i] = middle[i] - std_mult * std_dev
    
    return middle, upper, lower

@register
class TvExceededCandle(BaseStrategy):
    name = "tv_exceeded_candle"
    version = "1.0.0"
    description = "Exceeded Candle: Candle pattern breakout with Bollinger Band filtering"
    category = "multi_factor"
    tags = ["mean_reversion", "bollinger_bands", "candle_pattern"]

    def __init__(self, length: int = 20, mult: float = 2.0):
        self.length = length
        self.mult = mult

    def get_warmup_periods(self) -> int:
        return self.length + 5  # SMA period + buffer

    def generate_signal(self, window) -> Signal | None:
        df = window.historical()
        if df.shape[0] < self.get_warmup_periods():
            return None

        closes = df["close"].to_list()
        opens = df["open"].to_list()

        basis, upper, lower = bollinger_bands(closes, self.length, self.mult)

        current_close = closes[-1]
        current_open = opens[-1]

        # Candle color determination
        is_green = current_close > current_open
        is_red = current_close < current_open

        # Exceeded candle patterns
        green_exceeded = False
        red_exceeded = False

        if len(closes) > 1:
            prev_close = closes[-2]
            prev_open = opens[-2]

            green_exceeded = (
                (prev_close < prev_open) &  # Previous candle was red
                (current_close > current_open) &  # Current candle is green
                (current_close > prev_open)  # Current close > previous open
            )

            red_exceeded = (
                (prev_close > prev_open) &  # Previous candle was green
                (current_close < current_open) &  # Current candle is red
                (current_close < prev_open)  # Current close < previous open
            )

        # Last 2 out of 3 candles were red (used to avoid entries)
        last3red = False
        if len(closes) > 3:
            last3red = (opens[-3] > closes[-3]) and (opens[-4] > closes[-4])

        # Entry conditions
        if len(basis) > 0 and basis[-1] is not None:
            if green_exceeded and (current_close < basis[-1]) and (not last3red):
                return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"pattern": "green_exceeded"})
            # if red_exceeded and (current_close > basis[-1]): # Disabled in original
            #     return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"pattern": "red_exceeded"})

        # Exit conditions
        if len(upper) > 0 and upper[-1] is not None:
            if current_close > upper[-1]:
                return Signal(action="sell", strength=0.9, confidence=0.7, metadata={"exit": "upper_bb"})

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "length": {
                    "type": "integer",
                    "default": 20,
                    "description": "Bollinger Bands and SMA period"
                },
                "mult": {
                    "type": "number",
                    "default": 2.0,
                    "description": "Bollinger Bands standard deviation multiplier"
                }
            },
            "required": ["length", "mult"]
        }