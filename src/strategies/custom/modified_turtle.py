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

def atr(df: pl.DataFrame, period: int) -> pl.Series:
    """Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr0 = high - low
    tr1 = abs(high - close.shift(1).fill_null(0.0))
    tr2 = abs(low - close.shift(1).fill_null(0.0))

    tr = pl.max([tr0, tr1, tr2])
    return tr.rolling(window=period, min_periods=period).mean()

def volume_sma(df: pl.DataFrame, period: int) -> pl.Series:
    """Simple Moving Average of Volume."""
    return df["volume"].rolling(window=period, min_periods=period).mean()

@register
class ModifiedTurtle(BaseStrategy):
    name = "modified_turtle"
    version = "1.1.0"
    description = "Modified Turtle: N-day breakout/breakdown with volume confirmation and ATR trailing stop (long & short)"
    category = "momentum"
    tags = ["breakout", "momentum", "atr"]

    def __init__(self, high_period: int = 20, volume_mult: float = 1.5, atr_period: int = 14, atr_stop_mult: float = 2.0, min_price: float = 5.0, min_volume: float = 100_000):
        self.high_period = high_period
        self.volume_mult = volume_mult
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.min_price = min_price
        self.min_volume = min_volume

        self.highest_since_entry = None
        self.lowest_since_entry = None
        self.position_dir = None
        self.entry_price = None
        self.entry_date = None

    def get_warmup_periods(self) -> int:
        return self.high_period + 10

    def generate_signal(self, window) -> Signal | None:
        df = pl.concat([window.historical(), window.current_bar()])
        if len(df) < self.get_warmup_periods():
            return None

        current_bar = df.tail(1)
        close = current_bar["close"][0]
        high = current_bar["high"][0]
        low = current_bar["low"][0]
        volume = current_bar["volume"][0]

        high_n = df["high"].rolling(window=self.high_period, min_periods=self.high_period).max().to_list()[-2]
        low_n = df["low"].rolling(window=self.high_period, min_periods=self.high_period).min().to_list()[-2]
        volume_sma_n = df["volume"].rolling(window=self.high_period, min_periods=self.high_period).mean().to_list()[-2]

        if high_n is None or low_n is None or volume_sma_n is None:
            return None

        vol_confirm = volume > (self.volume_mult * volume_sma_n)
        meets_price = close >= self.min_price
        meets_volume = volume_sma_n >= self.min_volume
        base_filter = vol_confirm and meets_price and meets_volume

        # Long breakout
        breakout_up = close > high_n
        buy_mask = breakout_up and base_filter

        # Short breakdown
        breakout_down = close < low_n
        short_mask = breakout_down and base_filter

        if buy_mask:
            strength = min((close - high_n) / high_n, 1.0)
            return Signal(action="buy", strength=strength, confidence=1.0, metadata={"high_n": high_n, "volume_sma_n": volume_sma_n, "volume": volume, "close": close})
        elif short_mask:
            strength = min((low_n - close) / low_n, 1.0)
            return Signal(action="sell", strength=strength, confidence=1.0, metadata={"low_n": low_n, "volume_sma_n": volume_sma_n, "volume": volume, "close": close})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "high_period": {"type": "integer", "default": 20, "minimum": 1},
            "volume_mult": {"type": "number", "default": 1.5, "minimum": 0.1},
            "atr_period": {"type": "integer", "default": 14, "minimum": 1},
            "atr_stop_mult": {"type": "number", "default": 2.0, "minimum": 0.1},
            "min_price": {"type": "number", "default": 5.0, "minimum": 0.01},
            "min_volume": {"type": "integer", "default": 100_000, "minimum": 1000},
        }