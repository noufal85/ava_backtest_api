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

def ema(series: list[float], period: int) -> list[float]:
    """Calculates the Exponential Moving Average (EMA) of a series."""
    ema = [series[0]]  # Initialize EMA with the first value
    alpha = 2 / (period + 1)

    for i in range(1, len(series)):
        ema.append(alpha * series[i] + (1 - alpha) * ema[-1])
    return ema

def macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, list[float]]:
    """Calculates MACD, signal line, and histogram."""
    # Calculate EMAs
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)

    # MACD line is the difference between the fast and slow EMAs
    macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(closes))]

    # Signal line is the EMA of the MACD line
    signal_line = ema(macd_line, signal)

    # MACD histogram is the difference between the MACD line and the signal line
    # Pad signal_line with Nones to match the length of macd_line
    padding_length = len(macd_line) - len(signal_line)
    signal_line = [None] * padding_length + signal_line
    macd_histogram = [macd_line[i] - signal_line[i] if signal_line[i] is not None else None for i in range(len(macd_line))]

    return {"macd": macd_line, "signal": signal_line, "histogram": macd_histogram}

@register
class TvMacdHistogramReversal(BaseStrategy):
    name = "tv_macd_histogram_reversal"
    version = "1.0.0"
    description = "MACD Histogram Reversal: Long-only momentum reversal strategy"
    category = "trend"
    tags = ["macd", "histogram", "reversal"]

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, stop_loss_pct: float | None = None, take_profit_pct: float | None = None):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def get_parameter_schema(self) -> dict:
        return {
            "fast_period": {"type": "integer", "default": 12},
            "slow_period": {"type": "integer", "default": 26},
            "signal_period": {"type": "integer", "default": 9},
            "stop_loss_pct": {"type": ["number", "null"], "default": None},
            "take_profit_pct": {"type": ["number", "null"], "default": None},
        }

    def get_warmup_periods(self) -> int:
        return max(self.fast_period, self.slow_period, self.signal_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()
        
        if len(closes) < self.get_warmup_periods():
            return None

        macd_data = macd(closes, fast=self.fast_period, slow=self.slow_period, signal=self.signal_period)
        macd_histogram = macd_data["histogram"]
        
        if len(macd_histogram) < 3:
            return None

        hist = macd_histogram[-1]
        hist_1 = macd_histogram[-2]
        hist_2 = macd_histogram[-3]

        if hist is None or hist_1 is None or hist_2 is None:
            return None

        # Long signal: histogram turns up from down while negative
        if hist > hist_1 and hist_1 < hist_2 and hist < 0:
            return Signal(action="buy", strength=1.0, confidence=1.0)

        return None