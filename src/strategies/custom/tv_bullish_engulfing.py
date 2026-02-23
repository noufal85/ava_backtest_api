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

def calculate_rsi(closes: list[float], period: int = 14) -> list[float]:
    """Calculates the Relative Strength Index (RSI)."""
    if len(closes) < period + 1:
        return [np.nan] * len(closes)

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    
    avg_gain = np.mean([d for d in deltas[:period] if d > 0])
    avg_loss = np.mean([-d for d in deltas[:period] if d < 0])

    rsi_values = [np.nan] * period
    
    for i in range(period, len(closes) - 1):
        delta = closes[i+1] - closes[i]
        gain = delta if delta > 0 else 0
        loss = -delta if delta < 0 else 0

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)

    rsi_values = [np.nan] + rsi_values
    return rsi_values

@register
class TvBullishEngulfing(BaseStrategy):
    name = "tv_bullish_engulfing"
    version = "1.0.0"
    description = "Bullish Engulfing Pattern: Simple pattern recognition with RSI exit"
    category = "momentum"
    tags = ["pattern", "engulfing", "rsi"]

    def __init__(self, rsi_length: int = 2, rsi_exit_threshold: float = 90.0):
        self.rsi_length = rsi_length
        self.rsi_exit_threshold = rsi_exit_threshold

    def get_warmup_periods(self) -> int:
        return self.rsi_length + 5

    def generate_signal(self, window) -> Signal | None:
        df = window.historical()
        current_bar = window.current_bar()
        
        df = pl.concat([df, current_bar])
        
        closes = df["close"].to_list()
        opens = df["open"].to_list()

        if len(closes) < 2:
            return None

        rsi_values = calculate_rsi(closes, self.rsi_length)
        rsi = rsi_values[-1]
        
        # Bullish engulfing pattern detection
        prev_bearish = opens[-2] > closes[-2]
        close_exceeds_prev_open = closes[-1] > opens[-2]
        open_below_prev_close = opens[-1] < closes[-2]
        bullish_engulfing = prev_bearish and close_exceeds_prev_open and open_below_prev_close

        if bullish_engulfing:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"pattern": "bullish_engulfing"})
        
        if not np.isnan(rsi) and rsi > self.rsi_exit_threshold:
            return Signal(action="sell", strength=0.9, confidence=0.7, metadata={"reason": "rsi_exit"})

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "rsi_length": {
                    "type": "integer",
                    "default": 2,
                    "description": "RSI calculation period"
                },
                "rsi_exit_threshold": {
                    "type": "number",
                    "default": 90.0,
                    "description": "RSI exit threshold"
                }
            },
            "required": ["rsi_length", "rsi_exit_threshold"]
        }