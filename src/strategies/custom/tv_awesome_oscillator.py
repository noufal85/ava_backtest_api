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

def sma(series: pl.Series, period: int) -> pl.Series:
    """Simple Moving Average."""
    if len(series) < period:
        return pl.Series([None] * len(series))
    
    sma_values = []
    for i in range(len(series)):
        if i < period - 1:
            sma_values.append(None)
        else:
            window = series[i - period + 1:i + 1]
            sma_values.append(window.mean())
    return pl.Series(sma_values)

@register
class TvAwesomeOscillator(BaseStrategy):
    name = "tv_awesome_oscillator"
    version = "1.0.0"
    description = "Awesome Oscillator Strategy: Momentum strategy using AO crossovers and saucer patterns"
    category = "momentum"
    tags = ["momentum", "oscillator"]

    def __init__(self, ao_fast: int = 5, ao_slow: int = 34):
        self.ao_fast = ao_fast
        self.ao_slow = ao_slow

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "ao_fast": {
                    "type": "integer",
                    "default": 5,
                    "description": "Fast SMA period for Awesome Oscillator"
                },
                "ao_slow": {
                    "type": "integer",
                    "default": 34,
                    "description": "Slow SMA period for Awesome Oscillator"
                }
            },
        }

    def get_warmup_periods(self) -> int:
        return self.ao_slow + 5  # Longest period + buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.get_warmup_periods():
            return None

        # Calculate HL2
        hl2 = (df["high"] + df["low"]) / 2

        # Calculate Awesome Oscillator
        ao = sma(hl2, self.ao_fast) - sma(hl2, self.ao_slow)

        current_ao = ao[-1]
        prev_ao = ao[-2] if len(ao) > 1 else None
        prev2_ao = ao[-3] if len(ao) > 2 else None

        if prev_ao is None or prev2_ao is None:
            return None

        # AO crossovers
        ao_cross_above = (prev_ao <= 0) and (current_ao > 0)
        ao_cross_below = (prev_ao >= 0) and (current_ao < 0)

        # Saucer patterns
        saucer_long = (current_ao > 0) and (prev2_ao > prev_ao) and (current_ao > prev_ao)
        saucer_short = (current_ao < 0) and (prev2_ao < prev_ao) and (current_ao < prev_ao)

        if ao_cross_above or saucer_long:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"ao_value": current_ao})
        #if ao_cross_below or saucer_short:
        #    return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"ao_value": current_ao})

        return None