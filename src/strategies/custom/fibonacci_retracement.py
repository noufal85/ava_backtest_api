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

def rsi(closes: list[float], period: int = 14) -> list[float]:
    """Calculates the Relative Strength Index (RSI)."""
    if len(closes) < period + 1:
        return [np.nan] * len(closes)

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    seed_up = sum(d for d in deltas[:period] if d > 0) / period
    seed_down = -sum(d for d in deltas[:period] if d < 0) / period
    rs = seed_up / seed_down if seed_down != 0 else 0
    rsi = 100 - 100 / (1 + rs)
    rsi_values = [np.nan] * period + [rsi]

    for i in range(period, len(deltas)):
        delta = deltas[i]
        up = delta if delta > 0 else 0
        down = -delta if delta < 0 else 0
        seed_up = (seed_up * (period - 1) + up) / period
        seed_down = (seed_down * (period - 1) + down) / period
        rs = seed_up / seed_down if seed_down != 0 else 0
        rsi = 100 - 100 / (1 + rs)
        rsi_values.append(rsi)

    return rsi_values

def ema(closes: list[float], period: int = 50) -> list[float]:
    """Calculates the Exponential Moving Average (EMA)."""
    if len(closes) < period:
        return [np.nan] * len(closes)

    k = 2 / (period + 1)
    ema = [np.nan] * len(closes)
    ema[period - 1] = sum(closes[:period]) / period

    for i in range(period, len(closes)):
        ema[i] = closes[i] * k + ema[i - 1] * (1 - k)

    return ema

def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> list[float]:
    """Calculates the Average True Range (ATR)."""
    if len(highs) != len(lows) or len(highs) != len(closes):
        raise ValueError("Highs, lows, and closes lists must have the same length.")

    if len(highs) < 2:
        return [np.nan] * len(highs)

    tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
          if i > 0 else highs[i] - lows[i] for i in range(len(highs))]

    atr_values = [np.nan] * len(highs)
    atr_values[period - 1] = sum(tr[:period]) / period

    for i in range(period, len(highs)):
        atr_values[i] = (atr_values[i - 1] * (period - 1) + tr[i]) / period

    return atr_values

def volume_sma(volumes: list[float], period: int = 20) -> list[float]:
    """Calculates the Simple Moving Average (SMA) of volume."""
    if len(volumes) < period:
        return [np.nan] * len(volumes)

    sma_values = [np.nan] * len(volumes)
    for i in range(period - 1, len(volumes)):
        sma_values[i] = sum(volumes[i - period + 1:i + 1]) / period

    return sma_values

@register
class FibonacciRetracement(BaseStrategy):
    name = "fibonacci_retracement"
    version = "1.0.0"
    description = "Fibonacci retracement: enter at key levels after big moves"
    category = "multi_factor"
    tags = ["fibonacci", "retracement", "swing"]

    def __init__(
        self,
        swing_lookback: int = 40,
        min_swing_pct: float = 0.10,
        fib_tolerance: float = 0.02,
        rsi_period: int = 14,
        rsi_max: float = 60.0,
        ema_period: int = 50,
        vol_sma_period: int = 20,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        max_hold: int = 20,
    ):
        self.swing_lookback = swing_lookback
        self.min_swing_pct = min_swing_pct
        self.fib_tolerance = fib_tolerance
        self.rsi_period = rsi_period
        self.rsi_max = rsi_max
        self.ema_period = ema_period
        self.vol_sma_period = vol_sma_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.max_hold = max_hold

    def get_warmup_periods(self) -> int:
        return max(self.swing_lookback, self.rsi_period, self.ema_period, self.vol_sma_period, self.atr_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical().sort("timestamp")
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar]).sort("timestamp")

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        volumes = df["volume"].to_list()

        if len(closes) < self.get_warmup_periods():
            return None

        rsi_values = rsi(closes, self.rsi_period)
        ema_values = ema(closes, self.ema_period)
        atr_values = atr(highs, lows, closes, self.atr_period)
        vol_sma_values = volume_sma(volumes, self.vol_sma_period)

        df = df.with_columns([
            pl.Series(name="rsi", values=rsi_values),
            pl.Series(name="ema", values=ema_values),
            pl.Series(name="atr", values=atr_values),
            pl.Series(name="vol_sma", values=vol_sma_values),
        ])

        # Swing high/low over lookback (shifted to avoid lookahead)
        df = df.with_columns([
            pl.col("high").shift(1).rolling(window=self.swing_lookback, min_periods=self.swing_lookback).max().alias("swing_high"),
            pl.col("low").shift(1).rolling(window=self.swing_lookback, min_periods=self.swing_lookback).min().alias("swing_low"),
        ])

        df = df.with_columns([
            (pl.col("swing_high") - pl.col("swing_low")).alias("swing_range"),
            (pl.col("swing_range") / pl.col("swing_low")).alias("swing_pct"),
        ])

        # Fibonacci levels
        df = df.with_columns([
            (pl.col("swing_high") - 0.382 * pl.col("swing_range")).alias("fib_382"),
            (pl.col("swing_high") - 0.500 * pl.col("swing_range")).alias("fib_500"),
            (pl.col("swing_high") - 0.618 * pl.col("swing_range")).alias("fib_618"),
        ])

        current_row = df.tail(1)

        if current_row.is_empty() or current_row["swing_pct"][0] is None or math.isnan(current_row["swing_pct"][0]):
            return None

        big_swing = current_row["swing_pct"][0] > self.min_swing_pct

        fib_382 = current_row["fib_382"][0]
        fib_500 = current_row["fib_500"][0]
        fib_618 = current_row["fib_618"][0]
        current_close = current_row["close"][0]

        near_fib = (
            (abs(current_close - fib_382) / fib_382 < self.fib_tolerance if not math.isnan(fib_382) and fib_382 != 0 else False) or
            (abs(current_close - fib_500) / fib_500 < self.fib_tolerance if not math.isnan(fib_500) and fib_500 != 0 else False) or
            (abs(current_close - fib_618) / fib_618 < self.fib_tolerance if not math.isnan(fib_618) and fib_618 != 0 else False)
        )

        uptrend = current_close > current_row["ema"][0] if not math.isnan(current_row["ema"][0]) else False
        not_overbought = current_row["rsi"][0] < self.rsi_max if not math.isnan(current_row["rsi"][0]) else False

        if big_swing and near_fib and uptrend and not_overbought:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"strategy": "fibonacci_retracement"})

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "swing_lookback": {"type": "integer", "default": 40},
                "min_swing_pct": {"type": "number", "default": 0.10},
                "fib_tolerance": {"type": "number", "default": 0.02},
                "rsi_period": {"type": "integer", "default": 14},
                "rsi_max": {"type": "number", "default": 60.0},
                "ema_period": {"type": "integer", "default": 50},
                "vol_sma_period": {"type": "integer", "default": 20},
                "atr_period": {"type": "integer", "default": 14},
                "atr_stop_mult": {"type": "number", "default": 2.0},
                "max_hold": {"type": "integer", "default": 20},
            },
            "required": [
                "swing_lookback",
                "min_swing_pct",
                "fib_tolerance",
                "rsi_period",
                "rsi_max",
                "ema_period",
                "vol_sma_period",
                "atr_period",
                "atr_stop_mult",
                "max_hold",
            ],
        }