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

def calculate_rsi(prices: list[float], period: int) -> list[float]:
    """Calculates the Relative Strength Index (RSI) for a given list of prices."""
    if len(prices) < period + 1:
        return [np.nan] * len(prices)

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    
    avg_gain = np.mean([d for d in deltas[period-1:] if d > 0])
    avg_loss = np.mean([-d for d in deltas[period-1:] if d < 0])

    rsi_values = [np.nan] * period
    
    if avg_loss == 0:
        rsi_values.append(100.0)
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
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
    
    return rsi_values

@register
class TvDoubleRsi(BaseStrategy):
    name = "tv_double_rsi"
    version = "1.0.0"
    description = "Double RSI strategy with dual periods for mean reversion confirmation"
    category = "mean_reversion"
    tags = ["mean_reversion", "rsi"]

    def __init__(
        self,
        rsi_length: int = 14,
        rsi_length_secondary: int = 21,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        secondary_oversold: float = 35.0,
        secondary_overbought: float = 65.0,
        use_tp: bool = True,
        tp_percent: float = 1.2,
    ):
        self.rsi_length = rsi_length
        self.rsi_length_secondary = rsi_length_secondary
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.secondary_oversold = secondary_oversold
        self.secondary_overbought = secondary_overbought
        self.use_tp = use_tp
        self.tp_percent = tp_percent / 100.0

    def get_warmup_periods(self) -> int:
        return max(self.rsi_length, self.rsi_length_secondary) + 2

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])
        
        close_prices = df["close"].to_list()

        if len(close_prices) < self.get_warmup_periods():
            return None

        rsi_primary_values = calculate_rsi(close_prices, self.rsi_length)
        rsi_secondary_values = calculate_rsi(close_prices, self.rsi_length_secondary)

        rsi_primary = rsi_primary_values[-1]
        rsi_secondary = rsi_secondary_values[-1]
        rsi_primary_prev = rsi_primary_values[-2] if len(rsi_primary_values) > 1 else np.nan
        rsi_secondary_prev = rsi_secondary_values[-2] if len(rsi_secondary_values) > 1 else np.nan

        if math.isnan(rsi_primary_prev) or math.isnan(rsi_secondary_prev):
            return None

        long_signal = (rsi_primary_prev <= self.rsi_oversold) and (rsi_primary > self.rsi_oversold) and (rsi_secondary_prev < self.secondary_oversold)
        short_signal = (rsi_primary_prev >= self.rsi_overbought) and (rsi_primary < self.rsi_overbought) and (rsi_secondary_prev > self.secondary_overbought) and False # Disabled short

        if long_signal:
            return Signal(
                action="buy",
                strength=1.0,
                confidence=0.8,
                metadata={
                    "rsi_primary": rsi_primary,
                    "rsi_secondary": rsi_secondary,
                },
            )
        elif short_signal:
            return Signal(
                action="sell",
                strength=1.0,
                confidence=0.8,
                metadata={
                    "rsi_primary": rsi_primary,
                    "rsi_secondary": rsi_secondary,
                },
            )
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "rsi_length": {
                    "type": "integer",
                    "default": 14,
                    "description": "Primary RSI period",
                },
                "rsi_length_secondary": {
                    "type": "integer",
                    "default": 21,
                    "description": "Secondary RSI period",
                },
                "rsi_oversold": {
                    "type": "number",
                    "default": 30.0,
                    "description": "Oversold threshold for primary RSI",
                },
                "rsi_overbought": {
                    "type": "number",
                    "default": 70.0,
                    "description": "Overbought threshold for primary RSI",
                },
                "secondary_oversold": {
                    "type": "number",
                    "default": 35.0,
                    "description": "Oversold threshold for secondary RSI",
                },
                "secondary_overbought": {
                    "type": "number",
                    "default": 65.0,
                    "description": "Overbought threshold for secondary RSI",
                },
                "use_tp": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable take profit exits",
                },
                "tp_percent": {
                    "type": "number",
                    "default": 1.2,
                    "description": "Take profit percentage",
                },
            },
        }