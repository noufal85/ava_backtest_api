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

def _ma(series: list[float], period: int, ma_type: str = "EMA") -> list[float]:
    if ma_type == "EMA":
        alpha = 2 / (period + 1)
        ema = [series[0]]  # Initialize EMA with the first value
        for i in range(1, len(series)):
            ema.append(alpha * series[i] + (1 - alpha) * ema[-1])
        return ema
    elif ma_type == "SMA":
        sma = []
        for i in range(len(series)):
            if i < period - 1:
                sma.append(None)
            else:
                sma.append(sum(series[i - period + 1:i + 1]) / period)
        return sma
    elif ma_type == "WMA":
        wma = []
        weights = np.arange(1, period + 1, dtype=float)
        weights_sum = weights.sum()
        for i in range(len(series)):
            if i < period - 1:
                wma.append(None)
            else:
                window = series[i - period + 1:i + 1]
                wma.append(np.dot(window, weights) / weights_sum)
        return wma
    elif ma_type == "RMA":
        alpha = 1 / period
        rma = [series[0]]
        for i in range(1, len(series)):
            rma.append(alpha * series[i] + (1 - alpha) * rma[-1])
        return rma
    return _ma(series, period, "EMA")

@register
class DgMaCross(BaseStrategy):
    name = "dg_ma_cross"
    version = "1.0.0"
    description = "DillonGrech MA Cross: configurable fast/slow MA crossover"
    category = "trend"
    tags = ["ma", "crossover", "trend following"]

    def __init__(self, fast_period: int = 50, slow_period: int = 200,
                 fast_type: str = "EMA", slow_type: str = "EMA"):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_type = fast_type
        self.slow_type = slow_type

    def get_warmup_periods(self) -> int:
        return max(self.fast_period, self.slow_period) + 2

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()
        ma_fast_values = _ma(closes, self.fast_period, self.fast_type)
        ma_slow_values = _ma(closes, self.slow_period, self.slow_type)

        if len(ma_fast_values) < 3 or len(ma_slow_values) < 3:
            return None

        ma_fast = ma_fast_values[-1]
        ma_slow = ma_slow_values[-1]
        prev_fast = ma_fast_values[-2]
        prev_slow = ma_slow_values[-2]
        prev2_fast = ma_fast_values[-3]
        prev2_slow = ma_slow_values[-3]

        if prev_fast is None or prev_slow is None or prev2_fast is None or prev2_slow is None:
            return None

        if (prev_fast > prev_slow) and (prev2_fast <= prev2_slow):
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"ma_fast": ma_fast, "ma_slow": ma_slow})
        elif (prev_fast < prev_slow) and (prev2_fast >= prev2_slow):
            return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"ma_fast": ma_fast, "ma_slow": ma_slow})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "fast_period": {
                    "type": "integer",
                    "default": 50,
                    "description": "Fast MA period"
                },
                "slow_period": {
                    "type": "integer",
                    "default": 200,
                    "description": "Slow MA period"
                },
                "fast_type": {
                    "type": "string",
                    "enum": ["EMA", "SMA", "WMA", "RMA"],
                    "default": "EMA",
                    "description": "Fast MA type"
                },
                "slow_type": {
                    "type": "string",
                    "enum": ["EMA", "SMA", "WMA", "RMA"],
                    "default": "EMA",
                    "description": "Slow MA type"
                }
            },
            "required": ["fast_period", "slow_period", "fast_type", "slow_type"]
        }