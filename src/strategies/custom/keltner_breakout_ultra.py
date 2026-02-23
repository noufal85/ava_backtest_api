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
    tr = [0.0]  # First value is 0
    for i in range(1, len(high)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
    atr_values = [0.0] * len(high)
    atr_values[period - 1] = sum(tr[1:period]) / period  # Simple average for the first ATR value
    for i in range(period, len(high)):
        atr_values[i] = (atr_values[i - 1] * (period - 1) + tr[i]) / period
    return atr_values

def ema(data: list[float], period: int) -> list[float]:
    ema = [0.0] * len(data)
    multiplier = 2 / (period + 1)
    # Initialize EMA with the simple average of the first 'period' values
    ema[period - 1] = sum(data[:period]) / period
    # Calculate EMA for the rest of the values
    for i in range(period, len(data)):
        ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
    return ema

@register
class KeltnerBreakoutUltra(BaseStrategy):
    name = "keltner_breakout_ultra"
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
        risk_pct: float = 0.02,
    ):
        self.atr_period = atr_period
        self.base_period = base_period
        self.base_atr_mult = base_atr_mult
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.vol_mult = vol_mult
        self.risk_pct = risk_pct

    def get_parameter_schema(self) -> dict:
        return {
            "atr_period": {"type": "integer", "default": 14},
            "base_period": {"type": "integer", "default": 20},
            "base_atr_mult": {"type": "number", "default": 4.0},
            "ema_fast": {"type": "integer", "default": 10},
            "ema_slow": {"type": "integer", "default": 20},
            "vol_mult": {"type": "number", "default": 1.3},
            "risk_pct": {"type": "number", "default": 0.02},
        }

    def get_warmup_periods(self) -> int:
        return max(self.atr_period, self.base_period, self.ema_fast, self.ema_slow) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        volumes = df["volume"].to_list()

        if len(closes) < self.get_warmup_periods():
            return None

        atr_values = atr(highs, lows, closes, self.atr_period)
        ema_fast_values = ema(closes, self.ema_fast)
        ema_slow_values = ema(closes, self.ema_slow)

        # Base detection
        base_high = pl.Series(closes[:-1]).rolling_max(self.base_period, min_periods=self.base_period).to_list()
        base_low = pl.Series(closes[:-1]).rolling_min(self.base_period, min_periods=self.base_period).to_list()
        base_range = [(h - l) if h is not None and l is not None else 0.0 for h, l in zip(base_high, base_low)]
        avg_volume = pl.Series(volumes[:-1]).rolling_mean(self.base_period, min_periods=self.base_period).to_list()

        # Pad the lists to match the length of closes
        base_range = [0.0] * (len(closes) - len(base_range)) + base_range
        avg_volume = [0.0] * (len(closes) - len(avg_volume)) + avg_volume

        # Get previous values
        prev_close = closes[-2]
        prev_atr = atr_values[-2]
        prev_ema_fast = ema_fast_values[-2]
        prev_ema_slow = ema_slow_values[-2]
        prev_volume = volumes[-2]
        prev_avg_volume = avg_volume[-2]
        current_base_high = base_high[-1]
        current_base_range = base_range[-1]

        # Guard None values
        if any(v is None for v in [prev_atr, prev_ema_fast, prev_ema_slow, current_base_high, prev_avg_volume]):
            return None

        # Conditions
        in_base = current_base_range < (self.base_atr_mult * prev_atr)
        uptrend = prev_ema_fast > prev_ema_slow
        breakout = closes[-1] > current_base_high
        volume_confirm = volumes[-1] > (self.vol_mult * (avg_volume[-1] or 1))

        if in_base and uptrend and breakout and volume_confirm:
            return Signal(action="buy", strength=1.0, confidence=1.0)

        return None