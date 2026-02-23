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
    """Calculates the Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return np.nan

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    seed_up = sum(up for up in deltas[:period] if up > 0) / period
    seed_down = sum(abs(down) for down in deltas[:period] if down < 0) / period
    rs = seed_up / seed_down if seed_down != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    up_ema = seed_up
    down_ema = seed_down

    for i in range(period, len(deltas)):
        up = deltas[i] if deltas[i] > 0 else 0
        down = abs(deltas[i]) if deltas[i] < 0 else 0

        up_ema = (up * 1 / period) + up_ema * (1 - 1 / period)
        down_ema = (down * 1 / period) + down_ema * (1 - 1 / period)

        rs = up_ema / down_ema if down_ema != 0 else 0
        rsi = 100 - (100 / (1 + rs))

    return rsi

def ema(prices: list[float], period: int = 20) -> float:
    """Calculates the Exponential Moving Average (EMA)."""
    if len(prices) < period:
        return np.nan

    k = 2 / (period + 1)
    ema = prices[period - 1]  # Initialize EMA with the first data point
    for i in range(period, len(prices)):
        ema = (prices[i] * k) + ema * (1 - k)
    return ema

def sma(prices: list[float], period: int = 20) -> float:
    """Calculates the Simple Moving Average (SMA)."""
    if len(prices) < period:
        return np.nan
    return sum(prices[-period:]) / period

@register
class TvRsiEma(BaseStrategy):
    name = "tv_rsi_ema"
    version = "1.0.0"
    description = "RSI mean reversion with dual EMA trend filter"
    category = "multi_factor"
    tags = ["rsi", "ema", "mean_reversion", "trend"]

    def __init__(
        self,
        rsi_length: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        ma_type: str = "EMA",
        ma_length: int = 150,
        ma2_type: str = "EMA",
        ma2_length: int = 600,
    ):
        self.rsi_length = rsi_length
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.ma_type = ma_type
        self.ma_length = ma_length
        self.ma2_type = ma2_type
        self.ma2_length = ma2_length

    def get_parameter_schema(self) -> dict:
        return {
            "rsi_length": {"type": "integer", "default": 14},
            "rsi_overbought": {"type": "integer", "default": 70},
            "rsi_oversold": {"type": "integer", "default": 30},
            "ma_type": {"type": "string", "enum": ["SMA", "EMA"], "default": "EMA"},
            "ma_length": {"type": "integer", "default": 150},
            "ma2_type": {"type": "string", "enum": ["SMA", "EMA"], "default": "EMA"},
            "ma2_length": {"type": "integer", "default": 600},
        }

    def get_warmup_periods(self) -> int:
        return max(self.rsi_length, self.ma_length, self.ma2_length) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()

        if len(closes) < self.get_warmup_periods():
            return None

        current_rsi = rsi(closes, self.rsi_length)
        
        if self.ma_type == "SMA":
            current_ma = sma(closes, self.ma_length)
        else:
            current_ma = ema(closes, self.ma_length)
            
        if self.ma2_type == "SMA":
            current_ma2 = sma(closes, self.ma2_length)
        else:
            current_ma2 = ema(closes, self.ma2_length)

        if np.isnan(current_rsi) or np.isnan(current_ma) or np.isnan(current_ma2):
            return None

        # Entry signals
        if current_rsi < self.rsi_oversold and current_ma > current_ma2:
            return Signal(
                action="buy",
                strength=1.0,
                confidence=1.0,
                metadata={
                    "rsi": current_rsi,
                    "ma": current_ma,
                    "ma2": current_ma2,
                },
            )

        # Exit signals
        if current_rsi > self.rsi_overbought:
            return Signal(
                action="sell",
                strength=1.0,
                confidence=1.0,
                metadata={
                    "rsi": current_rsi,
                    "ma": current_ma,
                    "ma2": current_ma2,
                },
            )

        return None