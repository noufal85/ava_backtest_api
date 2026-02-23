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

def rsi(prices: list[float], period: int = 14) -> list[float]:
    """Calculates the Relative Strength Index (RSI) for a given list of prices."""
    if len(prices) < period + 1:
        return [np.nan] * len(prices)

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    avg_gain = np.mean([d for d in deltas[1:period+1] if d > 0])
    avg_loss = np.mean([-d for d in deltas[1:period+1] if d < 0])

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

def ema(prices: list[float], period: int) -> list[float]:
    """Calculates the Exponential Moving Average (EMA) for a given list of prices."""
    if len(prices) < period:
        return [np.nan] * len(prices)

    ema = [np.nan] * len(prices)
    ema[period - 1] = np.mean(prices[:period])
    k = 2 / (period + 1)

    for i in range(period, len(prices)):
        ema[i] = prices[i] * k + ema[i - 1] * (1 - k)

    return ema

def macd(prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[list[float], list[float], list[float]]:
    """Calculates the Moving Average Convergence Divergence (MACD) for a given list of prices."""
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)

    macd_line = [
        ema_fast[i] - ema_slow[i] if not math.isnan(ema_fast[i]) and not math.isnan(ema_slow[i]) else np.nan
        for i in range(len(prices))
    ]
    signal_line = ema([value for value in macd_line if not math.isnan(value)], signal)
    
    # Pad signal line with NaNs to match the length of macd_line
    padding_length = macd_line.index(next((x for x in macd_line if not math.isnan(x)), None)) if next((x for x in macd_line if not math.isnan(x)), None) is not None else 0
    signal_line = [np.nan] * padding_length + signal_line
    if len(signal_line) < len(macd_line):
        signal_line += [np.nan] * (len(macd_line) - len(signal_line))

    histogram = [
        macd_line[i] - signal_line[i] if not math.isnan(macd_line[i]) and not math.isnan(signal_line[i]) else np.nan
        for i in range(len(prices))
    ]

    return macd_line, signal_line, histogram

@register
class TvMacdRsi(BaseStrategy):
    name = "tv_macd_rsi"
    version = "1.0.0"
    description = "MACD + RSI Strategy: Combines MACD crossovers with RSI conditions"
    category = "mean_reversion"
    tags = ["macd", "rsi", "mean_reversion"]

    def __init__(
        self,
        fast_length: int = 12,
        slow_length: int = 26,
        signal_length: int = 9,
        rsi_length: int = 14,
        stop_loss_pct: float = 0.99,
        take_profit_pct: float = 10.0,
        rsi_back_candles: int = 5,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ):
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.signal_length = signal_length
        self.rsi_length = rsi_length
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.rsi_back_candles = rsi_back_candles
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "fast_length": {"type": "integer", "default": 12},
                "slow_length": {"type": "integer", "default": 26},
                "signal_length": {"type": "integer", "default": 9},
                "rsi_length": {"type": "integer", "default": 14},
                "stop_loss_pct": {"type": "number", "default": 0.99},
                "take_profit_pct": {"type": "number", "default": 10.0},
                "rsi_back_candles": {"type": "integer", "default": 5},
                "rsi_oversold": {"type": "number", "default": 30.0},
                "rsi_overbought": {"type": "number", "default": 70.0},
            },
        }

    def get_warmup_periods(self) -> int:
        return max(self.slow_length, self.rsi_length) + self.rsi_back_candles + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()

        # MACD calculation
        macd_line, signal_line, _ = macd(closes, fast=self.fast_length, slow=self.slow_length, signal=self.signal_length)
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        prev_macd = macd_line[-2] if len(macd_line) > 1 else np.nan
        prev_signal = signal_line[-2] if len(signal_line) > 1 else np.nan
        prev2_macd = macd_line[-3] if len(macd_line) > 2 else np.nan
        prev2_signal = signal_line[-3] if len(signal_line) > 2 else np.nan

        # RSI calculation
        rsi_values = rsi(closes, period=self.rsi_length)
        current_rsi = rsi_values[-1]

        # Check if RSI was oversold in last N candles
        was_rsi_oversold = False
        if len(rsi_values) > self.rsi_back_candles:
            for i in range(1, min(self.rsi_back_candles + 1, len(rsi_values))):
                if rsi_values[-1-i] <= self.rsi_oversold:
                    was_rsi_oversold = True
                    break

        # MACD crossovers
        bull_cross = (prev2_macd <= prev2_signal) and (prev_macd > prev_signal)

        # Entry conditions
        long_entry = bull_cross and was_rsi_oversold

        if long_entry:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"indicator": "macd_rsi"})

        return None