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

def sma(series: pl.Series, length: int) -> pl.Series:
    if len(series) < length:
        return pl.Series([None] * len(series))
    return series.rolling_mean(window_size=length, min_periods=length, center=False).alias("sma")

def rsi(series: pl.Series, length: int) -> pl.Series:
    if len(series) < length:
        return pl.Series([None] * len(series))

    delta = series.diff(n=1)
    up, down = delta.clone(), delta.clone()
    up = up.fill_null(0.0)
    down = down.fill_null(0.0)

    up = up.clip(lower_bound=0.0)
    down = down.clip(upper_bound=0.0).abs()

    avg_gain = up.rolling_mean(window_size=length, min_periods=length, center=False)
    avg_loss = down.rolling_mean(window_size=length, min_periods=length, center=False)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.alias("rsi")

def bollinger_bands(series: pl.Series, length: int, std: float) -> dict:
    middle_band = sma(series, length)
    if len(series) < length:
        upper_band = pl.Series([None] * len(series))
        lower_band = pl.Series([None] * len(series))
    else:
        rolling_std = series.rolling_std(window_size=length, min_periods=length, center=False)
        upper_band = middle_band + (rolling_std * std)
        lower_band = middle_band - (rolling_std * std)
    return {"upper": upper_band, "middle": middle_band, "lower": lower_band}

@register
class DgBbRsi(BaseStrategy):
    name = "dg_bb_rsi"
    version = "1.0.0"
    description = "DillonGrech BB+RSI mean reversion: buy below lower BB with oversold RSI, exit at BB center"
    category = "mean_reversion"
    tags = ["mean_reversion", "bollinger_bands", "rsi"]

    def __init__(self, bb_length: int = 20, bb_std: float = 2.0, rsi_length: int = 14, rsi_lower: float = 30.0):
        self.bb_length = bb_length
        self.bb_std = bb_std
        self.rsi_length = rsi_length
        self.rsi_lower = rsi_lower

    def get_parameter_schema(self) -> dict:
        return {
            "bb_length": {"type": "integer", "default": 20, "minimum": 2, "maximum": 100},
            "bb_std": {"type": "number", "default": 2.0, "minimum": 0.5, "maximum": 5.0},
            "rsi_length": {"type": "integer", "default": 14, "minimum": 2, "maximum": 50},
            "rsi_lower": {"type": "number", "default": 30.0, "minimum": 10.0, "maximum": 50.0},
        }

    def get_warmup_periods(self) -> int:
        return max(self.bb_length, self.rsi_length) + 2

    def generate_signal(self, window) -> Signal | None:
        df = pl.concat([window.historical(), window.current_bar()])
        if len(df) < self.get_warmup_periods():
            return None

        close = df["close"]
        bb = bollinger_bands(close, self.bb_length, self.bb_std)
        rsi_values = rsi(close, self.rsi_length)

        current_close = close[-1]
        current_bb_lower = bb["lower"][-1]
        current_bb_middle = bb["middle"][-1]
        current_rsi = rsi_values[-1]

        previous_close = close[-2]
        previous_bb_lower = bb["lower"][-2]
        previous_bb_middle = bb["middle"][-2]
        previous_rsi = rsi_values[-2]

        previous2_close = close[-3]
        previous2_bb_middle = bb["middle"][-3]

        if (previous_close < previous_bb_lower) and (previous_rsi < self.rsi_lower):
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"reason": "bb_rsi_oversold"})

        if (previous_close > previous_bb_middle) and (previous2_close <= previous2_bb_middle):
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"reason": "bb_center_cross"})

        return None