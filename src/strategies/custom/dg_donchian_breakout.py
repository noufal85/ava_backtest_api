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
class DgDonchianBreakout(BaseStrategy):
    name = "dg_donchian_breakout"
    version = "1.0.0"
    description = "DillonGrech Donchian New High/Low breakout with RR-based TP/SL"
    category = "momentum"
    tags = ["donchian", "breakout", "momentum"]

    def __init__(self, don_length: int = 20, profit_rr: float = 5.0):
        self.don_length = don_length
        self.profit_rr = profit_rr

    def get_warmup_periods(self) -> int:
        return self.don_length + 2

    def generate_signal(self, window) -> Signal | None:
        historical = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical, current_bar])

        length = self.don_length

        if len(df) < length + 2:
            return None

        high = df["high"].to_list()
        low = df["low"].to_list()
        close = df["close"].to_list()

        don_upper = [max(high[i - length + 1:i + 1]) if i >= length - 1 else None for i in range(len(high))]
        don_lower = [min(low[i - length + 1:i + 1]) if i >= length - 1 else None for i in range(len(low))]
        don_basis = [(don_upper[i] + don_lower[i]) / 2 if don_upper[i] is not None and don_lower[i] is not None else None for i in range(len(don_upper))]

        new_high = [True] * len(high)
        for i in range(length, len(high)):
            is_new_high = True
            for j in range(1, length):
                if don_upper[i] < don_upper[i-j]:
                    is_new_high = False
                    break
            new_high[i] = is_new_high

        prev_close = close[-2]
        prev2_close = close[-3]
        prev_don_upper = don_upper[-3]
        prev_new_high = new_high[-3]

        if prev_don_upper is None or not prev_new_high:
            return None

        if prev_close > prev_don_upper and prev2_close <= prev_don_upper:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"don_upper": prev_don_upper})

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "don_length": {
                "title": "Don Length",
                "type": "integer",
                "default": 20,
                "minimum": 1,
                "maximum": 100
            },
            "profit_rr": {
                "title": "Profit Rr",
                "type": "number",
                "default": 5.0,
                "minimum": 1.0,
                "maximum": 10.0
            }
        }