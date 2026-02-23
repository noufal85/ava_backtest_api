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

def rsi(closes, period=14):
    if len(closes) < period + 1:
        return np.nan

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    avg_gain = sum(d for d in deltas[:period] if d > 0) / period
    avg_loss = abs(sum(d for d in deltas[:period] if d < 0) / period)

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    for i in range(period, len(deltas)):
        gain = deltas[i] if deltas[i] > 0 else 0
        loss = abs(deltas[i]) if deltas[i] < 0 else 0

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        current_rsi = 100 - (100 / (1 + rs))
        rsi = current_rsi

    return rsi

def sma(closes, period=20):
    if len(closes) < period:
        return np.nan
    return sum(closes[-period:]) / period

@register
class ConnorsRsi(BaseStrategy):
    name = "connors_rsi"
    version = "1.1.0"
    description = "Mean reversion: RSI(2) oversold/overbought with SMA(200) trend filter (long & short)"
    category = "mean_reversion"
    tags = ["mean_reversion", "rsi", "sma"]

    def __init__(self, rsi_period: int = 2, entry_rsi: float = 10.0, exit_rsi: float = 90.0, sma_filter_period: int = 200):
        self.rsi_period = rsi_period
        self.entry_rsi = entry_rsi
        self.exit_rsi = exit_rsi
        self.sma_filter_period = sma_filter_period

    def get_warmup_periods(self) -> int:
        return self.sma_filter_period + 5 # SMA period + buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        combined_data = pl.concat([historical_data, current_bar])

        closes = combined_data["close"].to_list()
        
        if len(closes) < self.sma_filter_period:
            return None

        current_close = closes[-1]
        previous_close = closes[-2] if len(closes) > 1 else None

        closes_for_rsi = closes[- (self.rsi_period + 1):] if len(closes) > self.rsi_period else closes
        current_rsi = rsi(closes_for_rsi, self.rsi_period)
        
        closes_for_sma = closes[-self.sma_filter_period:]
        current_sma = sma(closes_for_sma, self.sma_filter_period)

        if current_rsi is np.nan or current_sma is np.nan or previous_close is None:
            return None

        if current_rsi < self.entry_rsi and previous_close > current_sma:
            strength = self.entry_rsi - current_rsi
            return Signal(action="buy", strength=strength / self.entry_rsi, confidence=1.0, metadata={"rsi": current_rsi, "sma": current_sma})
        elif current_rsi > self.exit_rsi and previous_close < current_sma:
            strength = current_rsi - self.exit_rsi
            return Signal(action="sell", strength=strength / (100 - self.exit_rsi), confidence=1.0, metadata={"rsi": current_rsi, "sma": current_sma})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "rsi_period": {
                    "type": "integer",
                    "default": 2,
                    "description": "RSI calculation period"
                },
                "entry_rsi": {
                    "type": "number",
                    "default": 10.0,
                    "description": "RSI threshold for long entry"
                },
                "exit_rsi": {
                    "type": "number",
                    "default": 90.0,
                    "description": "RSI threshold for long exit"
                },
                "sma_filter_period": {
                    "type": "integer",
                    "default": 200,
                    "description": "SMA period for trend filter"
                },
            },
        }