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

def rsi(closes: list[float], period: int = 14) -> list[float]:
    """Calculates the Relative Strength Index (RSI)."""
    if len(closes) < period + 1:
        return [np.nan] * len(closes)

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    avg_gain = np.mean([d for d in deltas[:period] if d > 0])
    avg_loss = np.mean([-d for d in deltas[:period] if d < 0])

    rsi_values = [np.nan] * period
    if avg_loss == 0:
        rsi_values.append(100.0 if avg_gain > 0 else 0.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs)))

    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain = delta if delta > 0 else 0
        loss = -delta if delta < 0 else 0

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi_value = 100.0 if avg_gain > 0 else 0.0
        else:
            rs = avg_gain / avg_loss
            rsi_value = 100 - (100 / (1 + rs))
        rsi_values.append(rsi_value)
    return rsi_values

def macd(closes: list[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple[list[float], list[float], list[float]]:
    """Calculates the Moving Average Convergence Divergence (MACD)."""
    def ema(data: list[float], period: int) -> list[float]:
        ema_values = [np.nan] * len(data)
        if len(data) < period:
            return ema_values
        
        ema_values[period - 1] = np.mean(data[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data)):
            ema_values[i] = (data[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]
        return ema_values

    macd_line = ema(closes, fast_period)
    slow_ema = ema(closes, slow_period)
    macd_line = [macd_line[i] - slow_ema[i] if not np.isnan(macd_line[i]) and not np.isnan(slow_ema[i]) else np.nan for i in range(len(closes))]
    signal_line = ema(macd_line, signal_period)
    histogram = [macd_line[i] - signal_line[i] if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]) else np.nan for i in range(len(closes))]
    return macd_line, signal_line, histogram

@register
class TvRsiMacdCombo(BaseStrategy):
    name = "tv_rsi_macd_combo"
    version = "1.0.0"
    description = "RSI MACD Combo: Combined RSI and MACD momentum strategy"
    category = "mean_reversion"
    tags = ["rsi", "macd", "momentum"]

    def __init__(self,
                 rsi_length: int = 14,
                 rsi_overbought: int = 70,
                 rsi_oversold: int = 30,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9):
        self.rsi_length = rsi_length
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def get_warmup_periods(self) -> int:
        return max(self.rsi_length, self.macd_slow, self.macd_signal) + 5

    def generate_signal(self, window: pl.DataFrame) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        combined_data = pl.concat([historical_data, current_bar])
        closes = combined_data["close"].to_list()

        if len(closes) < self.get_warmup_periods():
            return None

        rsi_values = rsi(closes, self.rsi_length)
        macd_line, signal_line, histogram = macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)

        current_rsi = rsi_values[-1]
        previous_rsi = rsi_values[-2]
        current_histogram = histogram[-1]

        if (previous_rsi <= self.rsi_oversold) and (current_rsi > self.rsi_oversold) and (current_histogram > 0):
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"rsi": current_rsi, "macd_histogram": current_histogram})

        if (previous_rsi >= self.rsi_overbought) and (current_rsi < self.rsi_overbought) and (current_histogram < 0):
            return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"rsi": current_rsi, "macd_histogram": current_histogram})

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "rsi_length": {"type": "integer", "default": 14, "title": "RSI Length"},
            "rsi_overbought": {"type": "integer", "default": 70, "title": "RSI Overbought"},
            "rsi_oversold": {"type": "integer", "default": 30, "title": "RSI Oversold"},
            "macd_fast": {"type": "integer", "default": 12, "title": "MACD Fast"},
            "macd_slow": {"type": "integer", "default": 26, "title": "MACD Slow"},
            "macd_signal": {"type": "integer", "default": 9, "title": "MACD Signal"},
        }