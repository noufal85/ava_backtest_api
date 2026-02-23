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

def sma(series: pl.Series, period: int) -> pl.Series:
    if len(series) < period:
        return pl.Series([None] * len(series))
    
    sma_values = []
    for i in range(len(series)):
        if i < period - 1:
            sma_values.append(None)
        else:
            window = series[i - period + 1:i + 1]
            sma_values.append(window.mean())
    return pl.Series(sma_values)

@register
class RsRotation(BaseStrategy):
    name = "rs_rotation"
    version = "1.1.0"
    description = "Relative Strength Rotation: long strong RS, short weak RS"
    category = "momentum"
    tags = ["momentum", "relative strength", "rotation"]

    def __init__(self, rebalance_freq: int = 21, lookback: int = 63, market_filter_sma: int = 200, short_rs_threshold: float = -0.05):
        self.rebalance_freq = rebalance_freq
        self.lookback = lookback
        self.market_filter_sma = market_filter_sma
        self.short_rs_threshold = short_rs_threshold

    def get_parameter_schema(self) -> dict:
        return {
            "rebalance_freq": {"type": "integer", "default": 21, "description": "Rebalance frequency"},
            "lookback": {"type": "integer", "default": 63, "description": "Lookback period for RS calculation"},
            "market_filter_sma": {"type": "integer", "default": 200, "description": "SMA period for market filter"},
            "short_rs_threshold": {"type": "number", "default": -0.05, "description": "RS return below this triggers short"},
        }

    def get_warmup_periods(self) -> int:
        return max(self.lookback, self.market_filter_sma) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < max(self.lookback, self.market_filter_sma):
            return None

        close_prices = df["close"]
        rs_return = close_prices.pct_change(periods=self.lookback)
        sma_values = sma(close_prices, self.market_filter_sma)

        current_rs = rs_return[-1]
        previous_close = close_prices[-2]
        previous_sma = sma_values[-2]

        if previous_sma is None:
            return None

        bull_market = previous_close > previous_sma
        bear_market = previous_close < previous_sma
        positive_rs = current_rs > 0
        weak_rs = current_rs < self.short_rs_threshold

        index = len(historical_data) # Index of current bar in combined df

        if index % self.rebalance_freq == 0:
            if bull_market and positive_rs:
                return Signal(action="buy", strength=float(current_rs), confidence=1.0)
            elif bear_market and weak_rs:
                return Signal(action="sell", strength=abs(float(current_rs)), confidence=1.0)
            else:
                return None
        else:
            return None