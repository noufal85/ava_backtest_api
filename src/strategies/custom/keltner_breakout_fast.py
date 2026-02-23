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

def atr(high: list[float], low: list[float], close: list[float], period: int) -> list[float]:
    tr = [0.0] * len(close)
    atr_values = [0.0] * len(close)

    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr_values[period - 1] = sum(tr[0:period]) / period
    for i in range(period, len(close)):
        atr_values[i] = (atr_values[i - 1] * (period - 1) + tr[i]) / period

    return atr_values

def ema(data: list[float], period: int) -> list[float]:
    ema_values = [0.0] * len(data)
    alpha = 2 / (period + 1)

    ema_values[period - 1] = sum(data[0:period]) / period
    for i in range(period, len(data)):
        ema_values[i] = (data[i] * alpha) + (ema_values[i - 1] * (1 - alpha))

    return ema_values

@register
class KeltnerBreakoutFast(BaseStrategy):
    name = "keltner_breakout_fast"
    version = "1.0.0"
    description = "Oliver Kell base-and-breakout: buy breakouts from tight consolidation bases"
    category = "momentum"
    tags = ["breakout", "momentum"]

    def __init__(
        self,
        atr_period: int = 14,
        base_period: int = 20,
        base_atr_mult: float = 4.0,
        ema_fast: int = 10,
        ema_slow: int = 20,
        vol_mult: float = 1.3,
        max_hold: int = 30,
        risk_pct: float = 0.02,
    ):
        self.atr_period = atr_period
        self.base_period = base_period
        self.base_atr_mult = base_atr_mult
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.vol_mult = vol_mult
        self.max_hold = max_hold
        self.risk_pct = risk_pct

    def get_parameter_schema(self) -> dict:
        return {
            "atr_period": {"type": "integer", "default": 14},
            "base_period": {"type": "integer", "default": 20},
            "base_atr_mult": {"type": "number", "default": 4.0},
            "ema_fast": {"type": "integer", "default": 10},
            "ema_slow": {"type": "integer", "default": 20},
            "vol_mult": {"type": "number", "default": 1.3},
            "max_hold": {"type": "integer", "default": 30},
            "risk_pct": {"type": "number", "default": 0.02},
        }

    def get_warmup_periods(self) -> int:
        return max(self.atr_period, self.base_period, self.ema_fast, self.ema_slow) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        close = df["close"].to_list()
        high = df["high"].to_list()
        low = df["low"].to_list()
        volume = df["volume"].to_list()

        if len(close) < self.get_warmup_periods():
            return None

        atr_values = atr(high, low, close, self.atr_period)
        ema_fast_values = ema(close, self.ema_fast)
        ema_slow_values = ema(close, self.ema_slow)

        base_high = [0.0] * len(close)
        base_low = [0.0] * len(close)
        base_range = [0.0] * len(close)
        avg_volume = [0.0] * len(close)

        for i in range(self.base_period, len(close)):
            base_high[i] = max(close[i - self.base_period:i])
            base_low[i] = min(close[i - self.base_period:i])
            base_range[i] = base_high[i] - base_low[i]
            avg_volume[i] = sum(volume[i - self.base_period:i]) / self.base_period

        i = len(close) - 1

        if i < self.base_period:
            return None

        prev_close = close[i-1]
        prev_atr = atr_values[i-1]
        prev_ema_fast = ema_fast_values[i-1]
        prev_ema_slow = ema_slow_values[i-1]
        prev_volume = volume[i-1]
        prev_avg_volume = avg_volume[i-1]

        in_base = base_range[i-1] < (self.base_atr_mult * prev_atr)
        uptrend = prev_ema_fast > prev_ema_slow
        breakout = close[i] > base_high[i-1]
        volume_confirm = volume[i] > (self.vol_mult * avg_volume[i-1])

        if in_base and uptrend and breakout and volume_confirm:
            return Signal(action="buy", strength=1.0, confidence=1.0)

        return None