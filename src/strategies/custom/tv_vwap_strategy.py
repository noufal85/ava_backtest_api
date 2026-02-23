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

def calculate_vwap(df: pl.DataFrame) -> pl.Series:
    """Calculates Volume Weighted Average Price (VWAP)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]
    vwap = (pv.cum_sum() / df["volume"].cum_sum()).alias("vwap")
    return vwap

def calculate_ema(df: pl.DataFrame, period: int) -> pl.Series:
    """Calculates Exponential Moving Average (EMA)."""
    close = df["close"].to_numpy()
    ema = np.zeros_like(close)
    alpha = 2 / (period + 1)

    ema[period - 1] = np.mean(close[:period])

    for i in range(period, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i - 1]

    return pl.Series("ema", ema)

@register
class TvVwapStrategy(BaseStrategy):
    name = "tv_vwap_strategy"
    version = "1.0.0"
    description = "VWAP Strategy: Volume Weighted Average Price with EMA filter"
    category = "volatility"
    tags = ["vwap", "ema", "trend"]

    def __init__(self, ema_length: int = 20):
        self.ema_length = ema_length

    def get_warmup_periods(self) -> int:
        return self.ema_length + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.ema_length:
            return None

        vwap_values = calculate_vwap(df)
        ema_values = calculate_ema(df, self.ema_length)

        df = df.with_columns([
            vwap_values,
            ema_values
        ])

        current_row = df.tail(1).to_dicts()[0]
        previous_row = df.slice(-2, 1).to_dicts()[0]

        curr_close = current_row["close"]
        curr_vwap = current_row["vwap"]
        curr_ema = current_row["ema"]
        prev_close = previous_row["close"]
        prev_vwap = previous_row["vwap"]

        # Long signal: price crosses above VWAP AND price > EMA
        price_cross_up_vwap = (prev_close <= prev_vwap) and (curr_close > curr_vwap)
        long_signal = price_cross_up_vwap and (curr_close > curr_ema)

        # Exit signal: price crosses below VWAP AND price < EMA
        price_cross_down_vwap = (prev_close >= prev_vwap) and (curr_close < curr_vwap)
        exit_signal = price_cross_down_vwap and (curr_close < curr_ema)

        if long_signal:
            return Signal(
                action="buy",
                strength=1.0,
                confidence=0.8,
                metadata={
                    "vwap_value": curr_vwap,
                    "ema_value": curr_ema,
                    "price": curr_close
                }
            )
        elif exit_signal:
            return Signal(
                action="sell",
                strength=1.0,
                confidence=0.8,
                metadata={
                    "vwap_value": curr_vwap,
                    "ema_value": curr_ema,
                    "price": curr_close
                }
            )
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "ema_length": {
                "title": "EMA Length",
                "type": "integer",
                "default": 20,
                "minimum": 2,
                "maximum": 200,
            }
        }