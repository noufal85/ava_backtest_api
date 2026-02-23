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

def _atr(high: list[float], low: list[float], close: list[float], period: int) -> list[float]:
    tr = [0.0] * len(close)
    atr = [0.0] * len(close)

    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr[period - 1] = sum(tr[period - 1::-1]) / period if period <= len(close) else 0.0

    for i in range(period, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr

def _ema(data: list[float], period: int) -> list[float]:
    ema = [0.0] * len(data)
    if len(data) < period:
        return ema
    
    ema[period - 1] = sum(data[period - 1::-1]) / period

    multiplier = 2 / (period + 1)
    for i in range(period, len(data)):
        ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
    return ema

def _keltner_channels(df: pl.DataFrame, ema_period: int, atr_period: int, atr_mult: float) -> pl.DataFrame:
    close = df["close"].to_list()
    high = df["high"].to_list()
    low = df["low"].to_list()

    ema_middle = _ema(close, ema_period)
    atr_val = _atr(high, low, close, atr_period)

    kc_upper = [ema_middle[i] + atr_mult * atr_val[i] if i >= atr_period and i >= ema_period else 0.0 for i in range(len(close))]
    kc_lower = [ema_middle[i] - atr_mult * atr_val[i] if i >= atr_period and i >= ema_period else 0.0 for i in range(len(close))]

    return pl.DataFrame({
        "kc_upper": kc_upper,
        "kc_middle": ema_middle,
        "kc_lower": kc_lower,
    })

@register
class KeltnerReversionUltra(BaseStrategy):
    name = "keltner_reversion_ultra"
    version = "1.1.0"
    description = "Mean reversion: buy below lower / short above upper Keltner channel"
    category = "mean_reversion"
    tags = ["mean_reversion", "keltner_channel"]

    def __init__(self, ema_period: int = 8, atr_period: int = 8, atr_mult: float = 1.5, max_hold: int = 5):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.max_hold = max_hold

    def get_parameter_schema(self) -> dict:
        return {
            "ema_period": {"type": "integer", "default": self.ema_period, "minimum": 2, "maximum": 50},
            "atr_period": {"type": "integer", "default": self.atr_period, "minimum": 2, "maximum": 50},
            "atr_mult": {"type": "number", "default": self.atr_mult, "minimum": 0.5, "maximum": 5.0},
            "max_hold": {"type": "integer", "default": self.max_hold, "minimum": 1, "maximum": 30}
        }

    def get_warmup_periods(self) -> int:
        return max(self.ema_period, self.atr_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_df = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_df, current_bar])

        if len(df) < max(self.ema_period, self.atr_period):
            return None

        kc = _keltner_channels(df, self.ema_period, self.atr_period, self.atr_mult)
        kc_upper = kc["kc_upper"].to_list()
        kc_middle = kc["kc_middle"].to_list()
        kc_lower = kc["kc_lower"].to_list()
        close = df["close"].to_list()

        prev_close = close[-2]
        prev_upper = kc_upper[-2]
        prev_lower = kc_lower[-2]
        prev_middle = kc_middle[-2]

        if prev_close < prev_lower:
            strength = prev_lower - prev_close
            return Signal(action="buy", strength=strength / prev_lower if prev_lower != 0 else 0.0, confidence=1.0, metadata={"prev_close": prev_close, "prev_middle": prev_middle})
        elif prev_close > prev_upper:
            strength = prev_close - prev_upper
            return Signal(action="sell", strength=strength / prev_close if prev_close != 0 else 0.0, confidence=1.0, metadata={"prev_close": prev_close, "prev_middle": prev_middle})
        else:
            return None