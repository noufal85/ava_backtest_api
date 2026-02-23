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
class TvRocMomentum(BaseStrategy):
    name = "tv_roc_momentum"
    version = "1.0.0"
    description = "ROC Momentum: Rate of Change momentum crossover strategy"
    category = "momentum"
    tags = ["momentum", "roc"]

    def __init__(self, roc_length: int = 12, signal_length: int = 9):
        self.roc_length = roc_length
        self.signal_length = signal_length

    def get_parameter_schema(self) -> dict:
        return {
            "roc_length": {"type": "integer", "default": 12, "minimum": 1},
            "signal_length": {"type": "integer", "default": 9, "minimum": 1},
        }

    def get_warmup_periods(self) -> int:
        return self.roc_length + self.signal_length + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.roc_length + self.signal_length:
            return None

        closes = df["close"]
        
        # Calculate Rate of Change (ROC)
        roc_values = []
        for i in range(len(closes)):
            if i < self.roc_length:
                roc_values.append(None)
            else:
                roc = ((closes[i] / closes[i - self.roc_length]) - 1) * 100
                roc_values.append(roc)
        roc_series = pl.Series(roc_values)
        
        # Calculate signal line as SMA of ROC
        roc_signal = sma(roc_series, period=self.signal_length)

        # Use shifted values to avoid look-ahead bias
        if len(roc_series) < 2 or len(roc_signal) < 2:
            return None

        prev_roc = roc_series[-2]
        prev_signal = roc_signal[-2]
        curr_roc = roc_series[-1]
        curr_signal = roc_signal[-1]
        
        # Long signal: ROC crosses above signal and ROC > 0
        roc_cross_up = (prev_roc is not None and prev_signal is not None and curr_roc is not None and curr_signal is not None and prev_roc <= prev_signal and curr_roc > curr_signal and curr_roc > 0)
        
        # Short/Exit signal: ROC crosses below signal and ROC < 0
        roc_cross_down = (prev_roc is not None and prev_signal is not None and curr_roc is not None and curr_signal is not None and prev_roc >= prev_signal and curr_roc < curr_signal and curr_roc < 0)
        
        if roc_cross_up:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"roc_value": curr_roc, "roc_signal": curr_signal})
        elif roc_cross_down:
            return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"roc_value": curr_roc, "roc_signal": curr_signal})
        else:
            return None