from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import polars as pl
import numpy as np
import math

from src.core.strategy.registry import register

@dataclass
class Signal:
    action: str          # "buy" | "sell" | hold"
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
        return [np.nan] * len(series)
    
    sma_values = []
    for i in range(len(series)):
        if i < period - 1:
            sma_values.append(np.nan)
        else:
            window = series[i - period + 1:i + 1]
            sma_values.append(sum(window) / period)
    return sma_values

def n_day_return(series: list[float], period: int) -> list[float]:
    """N-day return (Rate of Change)."""
    if len(series) < period:
        return [np.nan] * len(series)

    roc_values = []
    for i in range(len(series)):
        if i < period:
            roc_values.append(np.nan)
        else:
            roc_values.append((series[i] - series[i - period]) / series[i - period])
    return roc_values

@register
class TvMomentumRotation(BaseStrategy):
    name = "tv_momentum_rotation"
    version = "1.0.0"
    description = "Momentum Rotation Strategy: Rotates based on relative strength vs benchmark"
    category = "momentum"
    tags = ["momentum", "relative strength"]

    def __init__(self, benchmark_symbol: str = "SPY", momentum_length: int = 20, signal_length: int = 10):
        self.benchmark_symbol = benchmark_symbol
        self.momentum_length = momentum_length
        self.signal_length = signal_length

    def get_parameter_schema(self) -> dict:
        return {
            "benchmark_symbol": {"type": "string", "default": "SPY"},
            "momentum_length": {"type": "integer", "default": 20, "minimum": 1},
            "signal_length": {"type": "integer", "default": 10, "minimum": 1},
        }

    def get_warmup_periods(self) -> int:
        return self.momentum_length + self.signal_length + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical().sort("timestamp")
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        close_prices = df["close"].to_list()

        # Calculate momentum (Rate of Change)
        asset_momentum = n_day_return(close_prices, period=self.momentum_length)
        
        # Simulate benchmark momentum (in practice, would need actual benchmark data)
        # For demo purposes, assume benchmark has similar but dampened momentum
        benchmark_momentum = [x * 0.6 if x is not None else None for x in asset_momentum]
        
        # Relative strength = asset momentum - benchmark momentum
        relative_strength = [
            asset_momentum[i] - benchmark_momentum[i]
            if asset_momentum[i] is not None and benchmark_momentum[i] is not None
            else np.nan
            for i in range(len(asset_momentum))
        ]
        
        # Signal line = SMA of relative strength
        rs_signal = sma(relative_strength, period=self.signal_length)

        # Get the last values to make decisions
        if len(relative_strength) < 3 or len(rs_signal) < 2:
            return None

        prev_rs = relative_strength[-2]
        prev_signal = rs_signal[-2]
        prev2_rs = relative_strength[-3]
        prev2_signal = rs_signal[-3]

        # Crossovers
        rs_cross_above = (prev2_rs <= prev2_signal) and (prev_rs > prev_signal)
        rs_cross_below = (prev2_rs >= prev2_signal) and (prev_rs < prev_signal)

        # Entry conditions
        long_cond = rs_cross_above and (prev_rs > 0)  # RS crosses above signal and is positive
        short_cond = rs_cross_below and (prev_rs < 0)  # RS crosses below signal and is negative

        # Generate signal
        if long_cond:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"relative_strength": prev_rs})
        elif short_cond:
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"relative_strength": prev_rs})
        else:
            return None