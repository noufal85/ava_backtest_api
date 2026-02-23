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

def ema(series: list[float], period: int) -> list[float]:
    """Calculates Exponential Moving Average (EMA)."""
    ema = [np.nan] * len(series)
    if len(series) < period:
        return ema
    
    multiplier = 2 / (period + 1)
    ema[period-1] = sum(series[:period]) / period
    
    for i in range(period, len(series)):
        ema[i] = (series[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def atr(high: list[float], low: list[float], close: list[float], period: int) -> list[float]:
    """Calculates Average True Range (ATR)."""
    tr = [0.0] * len(high)
    atr_values = [np.nan] * len(high)

    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    atr_values[period-1] = sum(tr[1:period+1]) / period

    multiplier = 2 / (period + 1)
    for i in range(period, len(high)):
        atr_values[i] = (tr[i] - atr_values[i-1]) * multiplier + atr_values[i-1]

    return atr_values

@register
class TvKeltnerTrend(BaseStrategy):
    name = "tv_keltner_trend"
    version = "1.0.0"
    description = "Keltner Trend: Trend following with Keltner Channel breakouts"
    category = "trend"
    tags = ["keltner", "trend following", "breakout"]

    def __init__(self, ema_length: int = 20, atr_length: int = 10, multiplier: float = 2.0):
        self.ema_length = ema_length
        self.atr_length = atr_length
        self.multiplier = multiplier

    def get_parameter_schema(self) -> dict:
        return {
            "ema_length": {"type": "integer", "default": 20, "minimum": 1},
            "atr_length": {"type": "integer", "default": 10, "minimum": 1},
            "multiplier": {"type": "number", "default": 2.0, "minimum": 0.1},
        }

    def get_warmup_periods(self) -> int:
        return max(self.ema_length, self.atr_length) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()

        kc_basis = ema(closes, self.ema_length)
        atr_values = atr(highs, lows, closes, self.atr_length)

        if (
            math.isnan(kc_basis[-1])
            or math.isnan(atr_values[-1])
        ):
            return None

        kc_upper = kc_basis[-1] + self.multiplier * atr_values[-1]
        kc_lower = kc_basis[-1] - self.multiplier * atr_values[-1]

        if len(closes) < 2:
            return None

        prev_close = closes[-2]

        if closes[-1] > kc_upper and prev_close <= kc_upper:
            return Signal(
                action="buy",
                strength=1.0,
                confidence=0.8,
                metadata={
                    "kc_upper": kc_upper,
                    "kc_lower": kc_lower,
                    "kc_basis": kc_basis[-1],
                },
            )
        # Removed short signal for simplicity
        # if closes[-1] < kc_lower and prev_close >= kc_lower:
        #     return Signal(
        #         action="sell",
        #         strength=1.0,
        #         confidence=0.8,
        #         metadata={
        #             "kc_upper": kc_upper,
        #             "kc_lower": kc_lower,
        #             "kc_basis": kc_basis[-1],
        #         },
        #     )

        return None