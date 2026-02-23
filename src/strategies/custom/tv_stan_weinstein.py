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

def sma(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return [None] * len(values)
    
    sma_values = []
    for i in range(len(values)):
        if i < period - 1:
            sma_values.append(None)
        else:
            window = values[i - period + 1 : i + 1]
            sma_values.append(sum(window) / period)
    return sma_values

def n_day_high(highs: list[float], period: int) -> list[float]:
    if len(highs) < period:
        return [None] * len(highs)

    highest_highs = []
    for i in range(len(highs)):
        if i < period - 1:
            highest_highs.append(None)
        else:
            window = highs[i - period + 1 : i + 1]
            highest_highs.append(max(window))
    return highest_highs

@register
class TvStanWeinstein(BaseStrategy):
    name = "tv_stan_weinstein"
    version = "1.0.0"
    description = "Stan Weinstein Stage 2: Breakout strategy with relative strength and volume confirmation"
    category = "multi_factor"
    tags = ["trend", "breakout"]

    def __init__(
        self,
        rs_period: int = 50,
        volume_ma_length: int = 5,
        price_ma_length: int = 30,
        highest_lookback: int = 52,
        comparative_symbol: str = "SPY",
    ):
        self.rs_period = rs_period
        self.volume_ma_length = volume_ma_length
        self.price_ma_length = price_ma_length
        self.highest_lookback = highest_lookback
        self.comparative_symbol = comparative_symbol

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "rs_period": {"type": "integer", "default": 50},
                "volume_ma_length": {"type": "integer", "default": 5},
                "price_ma_length": {"type": "integer", "default": 30},
                "highest_lookback": {"type": "integer", "default": 52},
                "comparative_symbol": {"type": "string", "default": "SPY"},
            },
        }

    def get_warmup_periods(self) -> int:
        return max(self.rs_period, self.volume_ma_length, self.price_ma_length, self.highest_lookback) + 10

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical().sort("timestamp")
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar]).sort("timestamp")

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        volumes = df["volume"].to_list()

        volume_ma = sma(volumes, self.volume_ma_length)
        price_ma = sma(closes, self.price_ma_length)
        highest_high = n_day_high(highs, self.highest_lookback)

        rs_period = self.rs_period

        stock_return = [None] * len(closes)
        for i in range(rs_period, len(closes)):
            stock_return[i] = closes[i] / closes[i - rs_period] - 1

        spy_proxy_return = [x * 0.7 if x is not None else None for x in stock_return]
        rs_value = [stock_return[i] - spy_proxy_return[i] if stock_return[i] is not None and spy_proxy_return[i] is not None else None for i in range(len(stock_return))]

        current_index = len(closes) - 1

        if current_index < 1:
            return None

        prev_price_ma = price_ma[current_index - 1]
        prev_rs_value = rs_value[current_index - 1]
        prev_volume_ma = volume_ma[current_index - 1]
        prev_highest_high = highest_high[current_index - 1]

        if any(x is None for x in [prev_price_ma, prev_rs_value, prev_volume_ma, prev_highest_high]):
            return None

        price_above_ma = closes[current_index] > prev_price_ma
        rs_positive = prev_rs_value > 0
        volume_above_ma = volumes[current_index] > prev_volume_ma
        price_breakout = closes[current_index] > prev_highest_high

        if price_above_ma and rs_positive and volume_above_ma and price_breakout:
            return Signal(action="buy", strength=1.0, confidence=0.8)
        
        if closes[current_index] < prev_price_ma:
            return Signal(action="sell", strength=1.0, confidence=0.8)

        return None