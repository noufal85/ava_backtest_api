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
    atr_values = []
    for i in range(len(close)):
        if i < period:
            atr_values.append(None)
        else:
            tr_values = []
            for j in range(i - period + 1, i + 1):
                tr = max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1])) if j > 0 else high[j] - low[j]
                tr_values.append(tr)
            atr = sum(tr_values) / period
            atr_values.append(atr)
    return atr_values

def _ema(close: list[float], period: int) -> list[float]:
    ema_values = []
    k = 2 / (period + 1)
    ema = None
    for i, price in enumerate(close):
        if i < period - 1:
            ema_values.append(None)
        elif i == period - 1:
            ema = sum(close[:period]) / period
            ema_values.append(ema)
        else:
            ema = (price - ema) * k + ema
            ema_values.append(ema)
    return ema_values

def keltner_channels(df: pl.DataFrame, ema_period: int, atr_period: int, atr_mult: float) -> pl.DataFrame:
    close = df["close"].to_list()
    high = df["high"].to_list()
    low = df["low"].to_list()

    ema_values = _ema(close, ema_period)
    atr_values = _atr(high, low, close, atr_period)

    kc_upper = []
    kc_lower = []
    for i in range(len(close)):
        if ema_values[i] is None or atr_values[i] is None:
            kc_upper.append(None)
            kc_lower.append(None)
        else:
            kc_upper.append(ema_values[i] + atr_mult * atr_values[i])
            kc_lower.append(ema_values[i] - atr_mult * atr_values[i])

    return pl.DataFrame({
        "kc_upper": kc_upper,
        "kc_middle": ema_values,
        "kc_lower": kc_lower,
    })

@register
class KeltnerReversion(BaseStrategy):
    name = "keltner_reversion"
    version = "1.1.0"
    description = "Mean reversion: buy below lower / short above upper Keltner channel"
    category = "mean_reversion"
    tags = ["mean_reversion", "keltner_channel"]

    def __init__(self, ema_period: int = 20, atr_period: int = 14, atr_mult: float = 2.0, max_hold: int = 10):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.max_hold = max_hold

    def get_warmup_periods(self) -> int:
        return max(self.ema_period, self.atr_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < max(self.ema_period, self.atr_period):
            return None

        kc = keltner_channels(df, self.ema_period, self.atr_period, self.atr_mult)
        kc_upper = kc["kc_upper"].to_list()
        kc_middle = kc["kc_middle"].to_list()
        kc_lower = kc["kc_lower"].to_list()
        close = df["close"].to_list()

        prev_close = close[-2]
        prev_upper = kc_upper[-2]
        prev_lower = kc_lower[-2]
        prev_middle = kc_middle[-2]

        if prev_lower is None or prev_upper is None or prev_middle is None:
            return None

        if prev_close < prev_lower:
            strength = prev_lower - prev_close
            return Signal(action="buy", strength=strength, confidence=1.0, metadata={"prev_close": prev_close, "prev_middle": prev_middle})
        elif prev_close > prev_upper:
            strength = prev_close - prev_upper
            return Signal(action="sell", strength=strength, confidence=1.0, metadata={"prev_close": prev_close, "prev_middle": prev_middle})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "ema_period": {"type": "integer", "default": 20, "minimum": 2, "maximum": 200},
            "atr_period": {"type": "integer", "default": 14, "minimum": 2, "maximum": 200},
            "atr_mult": {"type": "number", "default": 2.0, "minimum": 0.1, "maximum": 5.0},
            "max_hold": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100}
        }