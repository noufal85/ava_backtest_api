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
        return [None] * len(values)
    
    sma_values = []
    for i in range(len(values)):
        if i < period - 1:
            sma_values.append(None)
        else:
            window = values[i - period + 1 : i + 1]
            sma_values.append(sum(window) / period)
    return sma_values

def n_day_high(highs: list[float], period: int) -> list[float]:
    if len(highs) < period:
        return [None] * len(highs)

    high_values = []
    for i in range(len(highs)):
        if i < period - 1:
            high_values.append(None)
        else:
            window = highs[i - period + 1 : i + 1]
            high_values.append(max(window))
    return high_values

def n_day_low(lows: list[float], period: int) -> list[float]:
    if len(lows) < period:
        return [None] * len(lows)

    low_values = []
    for i in range(len(lows)):
        if i < period - 1:
            low_values.append(None)
        else:
            window = lows[i - period + 1 : i + 1]
            low_values.append(min(window))
    return low_values

@register
class TvVolumeBreakout(BaseStrategy):
    name = "tv_volume_breakout"
    version = "1.0.0"
    description = "Volume Breakout: Long-only breakout strategy with volume confirmation"
    category = "multi_factor"
    tags = ["volume", "breakout", "long-only"]

    def __init__(
        self,
        volume_ma_period: int = 20,
        volume_multiplier: float = 2.0,
        price_breakout_period: int = 20,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ):
        self.volume_ma_period = volume_ma_period
        self.volume_multiplier = volume_multiplier
        self.price_breakout_period = price_breakout_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "volume_ma_period": {
                    "type": "integer",
                    "default": 20,
                    "description": "Volume moving average period",
                },
                "volume_multiplier": {
                    "type": "number",
                    "default": 2.0,
                    "description": "Volume spike threshold multiplier",
                },
                "price_breakout_period": {
                    "type": "integer",
                    "default": 20,
                    "description": "Lookback period for price breakouts",
                },
                "stop_loss_pct": {
                    "type": ["number", "null"],
                    "default": None,
                    "description": "Optional stop loss percentage",
                },
                "take_profit_pct": {
                    "type": ["number", "null"],
                    "default": None,
                    "description": "Optional take profit percentage",
                },
            },
        }

    def get_warmup_periods(self) -> int:
        return max(self.volume_ma_period, self.price_breakout_period) + 2

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical().sort("ts")
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar]).sort("ts")

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        volumes = df["volume"].to_list()

        volume_ma = sma(volumes, self.volume_ma_period)
        price_high = n_day_high(highs, self.price_breakout_period)
        price_low = n_day_low(lows, self.price_breakout_period)

        volume_spike = [
            v > (ma * self.volume_multiplier) if ma is not None else None
            for v, ma in zip(volumes, volume_ma)
        ]

        if (
            volume_spike[-2] is True
            and closes[-2] > price_high[-3] if price_high[-3] is not None else False
        ):
            return Signal(
                action="buy",
                strength=1.0,
                confidence=1.0,
                metadata={
                    "volume": volumes[-1],
                    "volume_ma": volume_ma[-1],
                    "volume_spike": volume_spike[-1],
                    "price_high": price_high[-1],
                },
            )

        return None