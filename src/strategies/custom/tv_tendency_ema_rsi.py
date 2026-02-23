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

def ema(series, period):
    alpha = 2 / (period + 1)
    ema = [series[0]]
    for i in range(1, len(series)):
        ema.append(alpha * series[i] + (1 - alpha) * ema[-1])
    return ema

def rsi(series, period):
    deltas = [series[i] - series[i-1] for i in range(1, len(series))]
    avg_gain = sum([d for d in deltas[:period] if d > 0]) / period
    avg_loss = abs(sum([d for d in deltas[:period] if d < 0]) / period)
    
    rsi_values = []
    for i in range(period, len(series)):
        delta = series[i] - series[i-1]
        gain = delta if delta > 0 else 0
        loss = abs(delta) if delta < 0 else 0
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
    
    return [None] * period + rsi_values

@register
class TvTendencyEmaRsi(BaseStrategy):
    name = "tv_tendency_ema_rsi"
    version = "1.0.0"
    description = "Tendency EMA RSI Strategy"
    category = "mean_reversion"
    tags = ["ema", "rsi", "mean_reversion"]

    def __init__(self, ema_period: int = 20, rsi_period: int = 14, rsi_oversold: int = 30, rsi_overbought: int = 70):
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def get_warmup_periods(self) -> int:
        return max(self.ema_period, self.rsi_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()

        if len(closes) < self.get_warmup_periods():
            return None

        ema_values = ema(closes, self.ema_period)
        rsi_values = rsi(closes, self.rsi_period)

        current_ema = ema_values[-1]
        current_rsi = rsi_values[-1]

        if current_rsi is None:
            return None

        if closes[-1] > current_ema and current_rsi < self.rsi_oversold:
            return Signal(action="buy", strength=1.0, confidence=0.8)
        elif closes[-1] < current_ema and current_rsi > self.rsi_overbought:
            return Signal(action="sell", strength=1.0, confidence=0.8)
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "ema_period": {"type": "integer", "default": 20, "minimum": 5, "maximum": 100},
            "rsi_period": {"type": "integer", "default": 14, "minimum": 2, "maximum": 50},
            "rsi_oversold": {"type": "integer", "default": 30, "minimum": 10, "maximum": 40},
            "rsi_overbought": {"type": "integer", "default": 70, "minimum": 60, "maximum": 90},
        }