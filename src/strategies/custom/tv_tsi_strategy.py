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
    """Calculates Exponential Moving Average (EMA) for a given series."""
    alpha = 2 / (period + 1)
    # Filter out leading Nones to find valid start
    valid = [(i, v) for i, v in enumerate(series) if v is not None]
    result = [None] * len(series)
    if len(valid) < period:
        return result
    # Use first `period` valid values for seed
    first_valid_idx = valid[0][0]
    seed_vals = [v for _, v in valid[:period]]
    seed_end = valid[period - 1][0]
    result[seed_end] = sum(seed_vals) / period
    prev = result[seed_end]
    for i in range(seed_end + 1, len(series)):
        if series[i] is not None and prev is not None:
            prev = alpha * series[i] + (1 - alpha) * prev
            result[i] = prev
        else:
            result[i] = prev
    return result

@register
class TvTsiStrategy(BaseStrategy):
    name = "tv_tsi_strategy"
    version = "1.0.0"
    description = "TSI Strategy: True Strength Index momentum oscillator strategy"
    category = "momentum"
    tags = ["momentum", "tradingview"]

    def __init__(self, long_length: int = 25, short_length: int = 13, signal_length: int = 13):
        self.long_length = long_length
        self.short_length = short_length
        self.signal_length = signal_length

    def get_warmup_periods(self) -> int:
        return max(self.long_length, self.short_length, self.signal_length) * 2

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()

        if len(closes) < max(self.long_length, self.short_length, self.signal_length):
            return None

        # Calculate price change (momentum)
        price_change = [closes[i] - closes[i-1] if i > 0 else 0 for i in range(len(closes))]
        abs_price_change = [abs(pc) for pc in price_change]

        # Apply double smoothing
        pc_smooth1 = ema(price_change, period=self.long_length)
        abs_pc_smooth1 = ema(abs_price_change, period=self.long_length)

        pc_smooth2 = ema(pc_smooth1, period=self.short_length)
        abs_pc_smooth2 = ema(abs_price_change, period=self.short_length)

        # Calculate TSI
        tsi = [100 * pc_smooth2[i] / abs_pc_smooth2[i] if pc_smooth2[i] is not None and abs_pc_smooth2[i] is not None and abs_pc_smooth2[i] != 0 else 0 for i in range(len(closes))]

        # Calculate signal line
        tsi_signal = ema(tsi, period=self.signal_length)

        # Use shifted values to avoid look-ahead bias
        if len(tsi) < 2 or len(tsi_signal) < 2:
            return None

        prev_tsi = tsi[-2]
        prev_signal = tsi_signal[-2]
        curr_tsi = tsi[-1]
        curr_signal = tsi_signal[-1]

        # Long signal: TSI crosses above signal
        tsi_cross_up = (prev_tsi <= prev_signal) and (curr_tsi > curr_signal)

        # Exit signal: TSI crosses below signal
        tsi_cross_down = (prev_tsi >= prev_signal) and (curr_tsi < curr_signal)

        if tsi_cross_up:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"tsi": curr_tsi, "tsi_signal": curr_signal})
        elif tsi_cross_down:
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"tsi": curr_tsi, "tsi_signal": curr_signal})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "long_length": {
                    "type": "integer",
                    "default": 25,
                    "description": "Long Length"
                },
                "short_length": {
                    "type": "integer",
                    "default": 13,
                    "description": "Short Length"
                },
                "signal_length": {
                    "type": "integer",
                    "default": 13,
                    "description": "Signal Length"
                },
            },
        }