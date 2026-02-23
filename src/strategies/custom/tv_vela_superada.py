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

def calculate_ema(series: pl.Series, period: int) -> pl.Series:
    """Calculates Exponential Moving Average (EMA) for a given series."""
    alpha = 2 / (period + 1)
    ema = [series[0]]  # Initialize EMA with the first value
    for i in range(1, len(series)):
        ema.append(alpha * series[i] + (1 - alpha) * ema[-1])
    return pl.Series(ema)

def calculate_rsi(series: pl.Series, period: int) -> pl.Series:
    """Calculates Relative Strength Index (RSI) for a given series."""
    deltas = series.diff().drop_nulls()
    seed = deltas[:period]
    up = seed.filter(seed >= 0).sum() / period
    down = -seed.filter(seed < 0).sum() / period
    rsis = [100 - 100 / (1 + up / down)]
    for i in range(period, len(deltas)):
        delta = deltas[i]
        if delta >= 0:
            up = (up * (period - 1) + delta) / period
            down = down * (period - 1) / period
        else:
            up = up * (period - 1) / period
            down = (down * (period - 1) - delta) / period
        rs = up / down
        rsi = 100 - 100 / (1 + rs)
        rsis.append(rsi)
    return pl.Series([None] * period + rsis)

def calculate_macd(series: pl.Series, fast_period: int, slow_period: int, signal_period: int) -> tuple[pl.Series, pl.Series, pl.Series]:
    """Calculates Moving Average Convergence Divergence (MACD) values."""
    ema_fast = calculate_ema(series, fast_period)
    ema_slow = calculate_ema(series, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

@register
class TvVelaSuperada(BaseStrategy):
    name = "tv_vela_superada"
    version = "1.0.0"
    description = "Vela Superada: Complex strategy with candle patterns, EMA, RSI, MACD and trailing stops"
    category = "multi_factor"
    tags = ["trend", "momentum"]

    def __init__(self, ema_length: int = 10, rsi_length: int = 14, tp_percent: float = 1.2, sl_percent: float = 1.8, show_long: bool = True, show_short: bool = False):
        self.ema_length = ema_length
        self.rsi_length = rsi_length
        self.tp_percent = tp_percent / 100
        self.sl_percent = sl_percent / 100
        self.show_long = show_long
        self.show_short = show_short

    def get_warmup_periods(self) -> int:
        return max(self.ema_length, self.rsi_length, 26) + 5 # Buffer

    def generate_signal(self, window) -> Signal | None:
        df = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([df, current_bar])

        if len(df) < self.get_warmup_periods():
            return None

        # Calculate indicators
        closes = df["close"]
        opens = df["open"]

        ema_values = calculate_ema(closes, self.ema_length)
        rsi_values = calculate_rsi(closes, self.rsi_length)
        macd_line, _, _ = calculate_macd(closes, 12, 26, 9)

        # Candle patterns (using lists for calculations)
        buy_pattern = (opens[-2] > closes[-2]) and (closes[-1] > opens[-1])
        sell_pattern = (opens[-2] < closes[-2]) and (closes[-1] < opens[-1])

        # Get previous values
        prev_ema = ema_values[-2]
        prev_rsi = rsi_values[-2]
        prev_macd = macd_line[-2]
        prev2_macd = macd_line[-3]
        prev_close = closes[-2]

        # Long entry conditions
        long_entry = (
            buy_pattern and
            (closes[-1] > prev_ema) and
            (prev_close > prev_ema) and
            (prev_rsi < 65) and
            (prev_macd > prev2_macd)
        )

        # Short entry conditions
        short_entry = (
            sell_pattern and
            (closes[-1] < prev_ema) and
            (prev_close < prev_ema) and
            (prev_rsi > 35) and
            (prev_macd < prev2_macd)
        )

        if self.show_long and long_entry:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"rsi": prev_rsi, "macd": prev_macd})

        if self.show_short and short_entry:
            return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"rsi": prev_rsi, "macd": prev_macd})

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "ema_length": {
                    "type": "integer",
                    "default": 10,
                    "description": "EMA length"
                },
                "rsi_length": {
                    "type": "integer",
                    "default": 14,
                    "description": "RSI length"
                },
                "tp_percent": {
                    "type": "number",
                    "default": 1.2,
                    "description": "Take profit percentage"
                },
                "sl_percent": {
                    "type": "number",
                    "default": 1.8,
                    "description": "Stop loss percentage"
                },
                "show_long": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable long entries"
                },
                "show_short": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable short entries"
                }
            }
        }