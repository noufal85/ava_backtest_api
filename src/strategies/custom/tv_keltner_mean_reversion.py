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

def calculate_ema(series: list[float], period: int) -> list[float]:
    """Calculates Exponential Moving Average (EMA) for a given series."""
    ema = [None] * len(series)
    multiplier = 2 / (period + 1)
    
    # Initialize EMA with SMA for the first 'period' values
    ema[period - 1] = sum(series[:period]) / period
    
    # Calculate EMA for the rest of the series
    for i in range(period, len(series)):
        ema[i] = (series[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    # Fill initial None values with the first valid EMA value
    first_valid_ema = next((x for x in ema if x is not None), None)
    for i in range(len(series)):
        if ema[i] is None:
            ema[i] = first_valid_ema
        else:
            break
    
    return ema

def calculate_atr(high: list[float], low: list[float], close: list[float], period: int) -> list[float]:
    """Calculates Average True Range (ATR) for a given period."""
    tr = [0.0] * len(high)
    atr = [None] * len(high)

    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr[period - 1] = sum(tr[1:period]) / period

    for i in range(period, len(high)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    first_valid_atr = next((x for x in atr if x is not None), None)
    for i in range(len(high)):
        if atr[i] is None:
            atr[i] = first_valid_atr
        else:
            break
    
    return atr

def calculate_rsi(close: list[float], period: int) -> list[float]:
    """Calculates Relative Strength Index (RSI) for a given period."""
    delta = [close[i] - close[i - 1] if i > 0 else 0 for i in range(len(close))]
    
    up, down = [0.0] * len(close), [0.0] * len(close)
    for i in range(1, len(close)):
        if delta[i] > 0:
            up[i] = delta[i]
        elif delta[i] < 0:
            down[i] = -delta[i]
    
    avg_gain = [0.0] * len(close)
    avg_loss = [0.0] * len(close)
    
    avg_gain[period] = sum(up[1:period+1]) / period
    avg_loss[period] = sum(down[1:period+1]) / period
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + up[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + down[i]) / period
    
    rs = [avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0 for i in range(len(close))]
    rsi = [100 - (100 / (1 + rs[i])) for i in range(len(close))]

    for i in range(period):
        rsi[i] = 50.0

    return rsi

def calculate_keltner_channels(close: list[float], high: list[float], low: list[float], period: int, multiplier: float) -> tuple[list[float], list[float], list[float]]:
    """Calculates Keltner Channels."""
    basis = calculate_ema(close, period)
    atr_val = calculate_atr(high, low, close, period)
    upper = [basis[i] + multiplier * atr_val[i] for i in range(len(close))]
    lower = [basis[i] - multiplier * atr_val[i] for i in range(len(close))]
    return upper, lower, basis

@register
class TvKeltnerMeanReversion(BaseStrategy):
    name = "tv_keltner_mean_reversion"
    version = "1.0.0"
    description = "Mean reversion strategy with Keltner Channels and RSI confirmation"
    category = "mean_reversion"
    tags = ["mean_reversion", "keltner_channel", "rsi"]

    def __init__(self, kc_length: int = 20, kc_mult: float = 2.0, rsi_length: int = 14, rsi_oversold: int = 30, rsi_overbought: int = 70):
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        self.rsi_length = rsi_length
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def get_parameter_schema(self) -> dict:
        return {
            "kc_length": {"type": "integer", "default": 20, "description": "Keltner Channel period"},
            "kc_mult": {"type": "number", "default": 2.0, "description": "Keltner Channel multiplier"},
            "rsi_length": {"type": "integer", "default": 14, "description": "RSI period"},
            "rsi_oversold": {"type": "integer", "default": 30, "description": "RSI oversold threshold"},
            "rsi_overbought": {"type": "integer", "default": 70, "description": "RSI overbought threshold"},
        }

    def get_warmup_periods(self) -> int:
        return max(self.kc_length, self.rsi_length) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()

        if historical_data is None or len(historical_data) < self.get_warmup_periods() or current_bar is None:
            return None

        combined_data = pl.concat([historical_data, current_bar])

        close = combined_data["close"].to_list()
        high = combined_data["high"].to_list()
        low = combined_data["low"].to_list()

        upper, lower, basis = calculate_keltner_channels(close, high, low, self.kc_length, self.kc_mult)
        rsi_values = calculate_rsi(close, self.rsi_length)

        current_close = close[-1]
        prev_close = close[-2]
        prev_upper = upper[-2]
        prev_lower = lower[-2]
        prev_basis = basis[-2]
        prev_rsi = rsi_values[-2]

        long_signal = (prev_close <= prev_lower) and (prev_rsi < self.rsi_oversold)
        short_signal = (prev_close >= prev_upper) and (prev_rsi > self.rsi_overbought) and False # Disabled for long-only

        long_exit = current_close >= prev_basis
        short_exit = current_close <= prev_basis

        if long_signal:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"strategy": self.name, "kc_lower": prev_lower, "rsi": prev_rsi})
        elif short_signal:
            return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"strategy": self.name, "kc_upper": prev_upper, "rsi": prev_rsi})
        elif long_exit:
            return Signal(action="sell", strength=0.9, confidence=0.7, metadata={"reason": "basis_return"})
        elif short_exit:
            return Signal(action="buy", strength=0.9, confidence=0.7, metadata={"reason": "basis_return"})
        else:
            return None