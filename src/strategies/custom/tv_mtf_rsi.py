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

@register
class TvMtfRsi(BaseStrategy):
    name = "tv_mtf_rsi"
    version = "1.0.0"
    description = "MtfRsi Strategy: Multi-timeframe strategy requiring higher timeframe data not available"
    category = "mean_reversion"
    tags = []

    def __init__(self):
        pass

    def get_warmup_periods(self) -> int:
        return 14  # RSI period + buffer

    def generate_signal(self, window: pl.DataFrame) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()

        if len(closes) < 14:
            return None

        def calculate_rsi(prices: list[float], period: int = 14) -> float:
            if len(prices) < period + 1:
                return 50.0  # Neutral value

            deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
            
            # Calculate average gains and losses over the period
            avg_gain = sum(delta for delta in deltas[-period:] if delta > 0) / period
            avg_loss = abs(sum(delta for delta in deltas[-period:] if delta < 0) / period)

            if avg_loss == 0:
                return 100.0 if avg_gain > 0 else 50.0  # Avoid division by zero

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        rsi = calculate_rsi(closes)

        if rsi < 30:
            return Signal(action="buy", strength=1.0, confidence=0.8)
        elif rsi > 70:
            return Signal(action="sell", strength=1.0, confidence=0.8)
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {},
        }