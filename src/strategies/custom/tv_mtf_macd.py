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

def ema(series: list[float], period: int) -> list[float]:
    alpha = 2 / (period + 1)
    ema = [series[0]]
    for i in range(1, len(series)):
        ema.append(alpha * series[i] + (1 - alpha) * ema[-1])
    return ema

def macd(series: list[float], fast_period: int, slow_period: int, signal_period: int) -> tuple[list[float], list[float], list[float]]:
    ema_fast = ema(series, fast_period)
    ema_slow = ema(series, slow_period)
    macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(series))]
    signal_line = ema(macd_line, signal_period)
    histogram = [macd_line[i] - signal_line[i] for i in range(len(series))]
    return macd_line, signal_line, histogram

@register
class TvMtfMacdStrategy(BaseStrategy):
    name = "tv_mtf_macd"
    version = "1.0.0"
    description = "MtfMacd Strategy: STUB - Multi-timeframe strategy requiring higher timeframe data not available"
    category = "trend"
    tags = ["tradingview", "macd", "mtf"]

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def get_warmup_periods(self) -> int:
        return max(self.fast_period, self.slow_period, self.signal_period) + 20

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])
        closes = df["close"].to_list()

        if len(closes) < max(self.fast_period, self.slow_period, self.signal_period):
            return None

        macd_line, signal_line, _ = macd(closes, self.fast_period, self.slow_period, self.signal_period)

        if len(macd_line) < 2:
            return None

        current_macd = macd_line[-1]
        previous_macd = macd_line[-2]
        current_signal = signal_line[-1]
        previous_signal = signal_line[-2]

        if previous_macd < previous_signal and current_macd > current_signal:
            return Signal(action="buy", strength=1.0, confidence=0.8)
        elif previous_macd > previous_signal and current_macd < current_signal:
            return Signal(action="sell", strength=1.0, confidence=0.8)
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "fast_period": {
                    "type": "integer",
                    "default": 12,
                    "minimum": 2,
                    "maximum": 50
                },
                "slow_period": {
                    "type": "integer",
                    "default": 26,
                    "minimum": 5,
                    "maximum": 100
                },
                "signal_period": {
                    "type": "integer",
                    "default": 9,
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["fast_period", "slow_period", "signal_period"]
        }