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

def sma(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return [np.nan] * len(values)
    
    sma_values = []
    for i in range(len(values)):
        if i < period - 1:
            sma_values.append(np.nan)
        else:
            window = values[i - period + 1 : i + 1]
            sma_values.append(sum(window) / period)
    return sma_values

def std_dev(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return [np.nan] * len(values)

    std_values = []
    for i in range(len(values)):
        if i < period - 1:
            std_values.append(np.nan)
        else:
            window = values[i - period + 1 : i + 1]
            std_values.append(np.std(window))
    return std_values

@register
class TvBollingerBreakout(BaseStrategy):
    name = "tv_bollinger_breakout"
    version = "1.0.0"
    description = "Bollinger Breakout: Trades breakouts of custom Bollinger Bands"
    category = "momentum"
    tags = ["bollinger", "breakout"]

    def __init__(self, sma_length: int = 350, std_length: int = 350, ub_offset: float = 2.5, lb_offset: float = 2.5, show_long: bool = True, show_short: bool = False):
        self.sma_length = sma_length
        self.std_length = std_length
        self.ub_offset = ub_offset
        self.lb_offset = lb_offset
        self.show_long = show_long
        self.show_short = show_short

    def get_parameter_schema(self) -> dict:
        return {
            "sma_length": {"type": "integer", "default": 350},
            "std_length": {"type": "integer", "default": 350},
            "ub_offset": {"type": "number", "default": 2.5},
            "lb_offset": {"type": "number", "default": 2.5},
            "show_long": {"type": "boolean", "default": True},
            "show_short": {"type": "boolean", "default": False},
        }

    def get_warmup_periods(self) -> int:
        return max(self.sma_length, self.std_length) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()

        sma_values = sma(closes, self.sma_length)
        std_devs = std_dev(closes, self.std_length)

        upper_bands = [sma_values[i] + (std_devs[i] * self.ub_offset) if not math.isnan(sma_values[i]) and not math.isnan(std_devs[i]) else np.nan for i in range(len(closes))]
        lower_bands = [sma_values[i] - (std_devs[i] * self.lb_offset) if not math.isnan(sma_values[i]) and not math.isnan(std_devs[i]) else np.nan for i in range(len(closes))]

        current_close = closes[-1]
        previous_close = closes[-2] if len(closes) > 1 else np.nan
        current_sma = sma_values[-1]
        previous_sma = sma_values[-2] if len(sma_values) > 1 else np.nan
        current_upper_band = upper_bands[-1]
        previous_upper_band = upper_bands[-2] if len(upper_bands) > 1 else np.nan
        current_lower_band = lower_bands[-1]
        previous_lower_band = lower_bands[-2] if len(lower_bands) > 1 else np.nan

        if not all(map(lambda x: not math.isnan(x), [previous_close, previous_upper_band, previous_lower_band, previous_sma, current_close, current_upper_band, current_lower_band, current_sma])):
            return None

        if self.show_long and previous_close <= previous_upper_band and current_close > current_upper_band:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"upper_band": current_upper_band, "lower_band": current_lower_band, "sma_value": current_sma})

        if self.show_short and previous_close >= previous_lower_band and current_close < current_lower_band:
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"upper_band": current_upper_band, "lower_band": current_lower_band, "sma_value": current_sma})

        if previous_close >= previous_sma and current_close < current_sma:
            return Signal(action="sell", strength=0.8, confidence=0.7, metadata={"reason": "sma_crossunder"})

        if previous_close <= previous_sma and current_close > current_sma:
            return Signal(action="buy", strength=0.8, confidence=0.7, metadata={"reason": "sma_crossover"})

        return None