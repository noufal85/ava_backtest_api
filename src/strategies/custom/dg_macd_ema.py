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

def calculate_ema(series: list[float], period: int) -> list[float]:
    ema = [None] * len(series)
    if len(series) < period:
        return ema
    
    multiplier = 2 / (period + 1)
    ema[period - 1] = sum(series[:period]) / period
    
    for i in range(period, len(series)):
        ema[i] = (series[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema

def calculate_macd(series: list[float], fast_period: int, slow_period: int, signal_period: int) -> tuple[list[float], list[float]]:
    ema_fast = calculate_ema(series, fast_period)
    ema_slow = calculate_ema(series, slow_period)
    
    macd_line = [
        ema_fast[i] - ema_slow[i] if ema_fast[i] is not None and ema_slow[i] is not None else None
        for i in range(len(series))
    ]
    
    signal_line = calculate_ema([x for x in macd_line if x is not None], signal_period)
    
    # Pad the signal line with Nones to match the length of macd_line
    padding_length = macd_line.index(next(x for x in macd_line if x is not None)) if any(x is not None for x in macd_line) else len(macd_line)
    signal_line = [None] * padding_length + signal_line if signal_line else [None] * len(macd_line)
    if len(signal_line) < len(macd_line):
        signal_line += [None] * (len(macd_line) - len(signal_line))
    
    return macd_line, signal_line

@register
class DgMacdEma(BaseStrategy):
    name = "dg_macd_ema"
    version = "1.0.0"
    description = "DillonGrech MACD+EMA: MACD cross below zero with EMA trend filter"
    category = "trend"
    tags = ["macd", "ema", "trend following"]

    def __init__(self, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9, ema_period: int = 365):
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_period = ema_period

    def get_warmup_periods(self) -> int:
        return max(self.ema_period, self.macd_slow) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()

        macd_line, macd_signal_line = calculate_macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        trend_ema = calculate_ema(closes, self.ema_period)

        if len(closes) < 3:
            return None

        prev_macd = macd_line[-2]
        prev_sig = macd_signal_line[-2]
        prev2_macd = macd_line[-3]
        prev2_sig = macd_signal_line[-3]
        prev_close = closes[-2]
        prev_ema = trend_ema[-2]
        prev2_close = closes[-3]
        prev2_ema = trend_ema[-3]

        if prev_macd is None or prev_sig is None or prev2_macd is None or prev2_sig is None or prev_close is None or prev_ema is None or prev2_close is None or prev2_ema is None:
            return None

        # MACD crosses above signal while both negative, price above EMA
        macd_cross_up = (prev_macd > prev_sig) and (prev2_macd <= prev2_sig) and (prev_macd < 0) and (prev_sig < 0)
        above_ema = prev_close > prev_ema

        # Exit: close crosses below EMA
        exit_long = (prev_close < prev_ema) and (prev2_close >= prev2_ema)

        if macd_cross_up and above_ema:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"reason": "macd_ema_cross"})
        elif exit_long:
            return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"reason": "ema_cross_below"})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "macd_fast": {
                    "type": "integer",
                    "default": 12,
                    "description": "Fast period for MACD calculation"
                },
                "macd_slow": {
                    "type": "integer",
                    "default": 26,
                    "description": "Slow period for MACD calculation"
                },
                "macd_signal": {
                    "type": "integer",
                    "default": 9,
                    "description": "Signal period for MACD calculation"
                },
                "ema_period": {
                    "type": "integer",
                    "default": 365,
                    "description": "Period for EMA calculation"
                },
            },
            "required": ["macd_fast", "macd_slow", "macd_signal", "ema_period"],
        }