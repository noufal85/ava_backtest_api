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

def rsi(prices: list[float], period: int = 14) -> list[float]:
    """Compute the Relative Strength Index (RSI) from a list of prices."""
    if len(prices) < period + 1:
        return [np.nan] * len(prices)

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    avg_gain = np.mean([d for d in deltas[:period] if d > 0])
    avg_loss = np.mean([-d for d in deltas[:period] if d < 0])

    rsi_values = [np.nan] * period
    if avg_loss == 0:
        rsi_values.append(100.0 if avg_gain > 0 else 0.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs)))

    for i in range(period + 1, len(prices)):
        delta = prices[i] - prices[i - 1]
        gain = delta if delta > 0 else 0
        loss = -delta if delta < 0 else 0

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi_val = 100.0 if avg_gain > 0 else 0.0
        else:
            rs = avg_gain / avg_loss
            rsi_val = 100 - (100 / (1 + rs))
        rsi_values.append(rsi_val)

    return rsi_values

@register
class OversoldBounceRsi(BaseStrategy):
    name = "oversold_bounce_rsi"
    version = "1.0.0"
    description = "Mean reversion: buy when RSI is oversold, exit on recovery"
    category = "mean_reversion"
    tags = ["rsi", "mean_reversion", "oversold"]

    def __init__(
        self,
        rsi_period: int = 14,
        entry_rsi: float = 30.0,
        exit_rsi: float = 50.0,
    ):
        self.rsi_period = rsi_period
        self.entry_rsi = entry_rsi
        self.exit_rsi = exit_rsi

    def get_warmup_periods(self) -> int:
        return self.rsi_period + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        combined_data = pl.concat([historical_data, current_bar])
        close_prices = combined_data["close"].to_list()
        
        if len(close_prices) < self.rsi_period + 1:
            return None

        rsi_values = rsi(close_prices, self.rsi_period)
        current_rsi = rsi_values[-1]
        previous_rsi = rsi_values[-2] if len(rsi_values) > 1 else None

        if previous_rsi is None or math.isnan(previous_rsi):
            return None

        if previous_rsi < self.entry_rsi:
            strength = self.entry_rsi - previous_rsi
            return Signal(action="buy", strength=strength / 100, confidence=1.0, metadata={"rsi": previous_rsi})

        if previous_rsi > self.exit_rsi:
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"rsi": previous_rsi})

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "rsi_period": {
                    "type": "integer",
                    "default": 14,
                    "description": "RSI calculation period",
                },
                "entry_rsi": {
                    "type": "number",
                    "default": 30.0,
                    "description": "RSI threshold for long entry",
                },
                "exit_rsi": {
                    "type": "number",
                    "default": 50.0,
                    "description": "RSI threshold for long exit",
                },
            },
        }