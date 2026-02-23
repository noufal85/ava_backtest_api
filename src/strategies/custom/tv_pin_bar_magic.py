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

def sma(series: pl.Series, period: int) -> pl.Series:
    """Simple Moving Average."""
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

def ema(series: pl.Series, period: int) -> pl.Series:
    """Exponential Moving Average."""
    if len(series) < period:
        return pl.Series([None] * len(series))

    alpha = 2 / (period + 1)
    ema_values = [None] * len(series)
    
    # Initialize EMA with SMA for the first 'period' values
    initial_sma = series[:period].mean()
    ema_values[period - 1] = initial_sma

    # Calculate EMA for the rest of the series
    for i in range(period, len(series)):
        ema_values[i] = (series[i] * alpha) + (ema_values[i - 1] * (1 - alpha))

    return pl.Series(ema_values)

def atr(df: pl.DataFrame, period: int) -> pl.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    true_range_values = []
    for i in range(len(df)):
        if i == 0:
            true_range_values.append(high[i] - low[i])
        else:
            prev_close = close[i - 1]
            high_low = high[i] - low[i]
            high_close = abs(high[i] - prev_close)
            low_close = abs(low[i] - prev_close)
            true_range_values.append(max(high_low, high_close, low_close))

    true_range = pl.Series(true_range_values)
    atr_values = [None] * len(df)

    if len(df) < period:
        return pl.Series(atr_values)

    # Initialize ATR with SMA of TR for the first 'period' values
    initial_atr = true_range[:period].mean()
    atr_values[period - 1] = initial_atr

    # Calculate ATR for the rest of the series
    for i in range(period, len(df)):
        atr_values[i] = (atr_values[i - 1] * (period - 1) + true_range[i]) / period

    return pl.Series(atr_values)

@register
class TvPinBarMagic(BaseStrategy):
    name = "tv_pin_bar_magic"
    version = "1.0.0"
    description = "Pin Bar Magic: Professional pin bar breakout with EMA fan filter"
    category = "momentum"
    tags = ["pin bar", "ema", "atr"]

    def __init__(
        self,
        equity_risk_pct: float = 0.03,
        atr_mult: float = 0.5,
        atr_period: int = 14,
        sma_slow: int = 50,
        ema_medium: int = 18,
        ema_fast: int = 6,
        cancel_entry_bars: int = 3,
        trail_points: int = 1,
        trail_offset: int = 1,
        pin_bar_ratio: float = 0.66,
    ):
        self.equity_risk_pct = equity_risk_pct
        self.atr_mult = atr_mult
        self.atr_period = atr_period
        self.sma_slow = sma_slow
        self.ema_medium = ema_medium
        self.ema_fast = ema_fast
        self.cancel_entry_bars = cancel_entry_bars
        self.trail_points = trail_points
        self.trail_offset = trail_offset
        self.pin_bar_ratio = pin_bar_ratio

    def get_warmup_periods(self) -> int:
        return max(self.atr_period, self.sma_slow, self.ema_medium, self.ema_fast) + 5

    def generate_signal(self, window) -> Signal | None:
        df = window.historical()
        if df is None or len(df) < self.get_warmup_periods():
            return None

        current_bar = window.current_bar()
        if current_bar is None:
            return None

        df = pl.concat([df, current_bar])

        # Moving averages
        slow_sma = sma(df["close"], period=self.sma_slow)
        medium_ema = ema(df["close"], period=self.ema_medium)
        fast_ema = ema(df["close"], period=self.ema_fast)
        atr_values = atr(df, period=self.atr_period)

        # Pin bar detection
        candle_range = df["high"] - df["low"]
        
        # Bullish pin bar: long lower wick (>= 66% of total range)
        bullish_condition1 = (df["close"] > df["open"]) & ((df["open"] - df["low"]) > self.pin_bar_ratio * candle_range)
        bullish_condition2 = (df["close"] < df["open"]) & ((df["close"] - df["low"]) > self.pin_bar_ratio * candle_range)
        bullish_pin_bar = bullish_condition1 | bullish_condition2

        # Bearish pin bar: long upper wick (>= 66% of total range)
        bearish_condition1 = (df["close"] > df["open"]) & ((df["high"] - df["close"]) > self.pin_bar_ratio * candle_range)
        bearish_condition2 = (df["close"] < df["open"]) & ((df["high"] - df["open"]) > self.pin_bar_ratio * candle_range)
        bearish_pin_bar = bearish_condition1 | bearish_condition2

        # EMA Fan trend conditions (using shifted values)
        if len(df) < 2:
            return None

        prev_fast_ema = fast_ema[-2]
        prev_medium_ema = medium_ema[-2]
        prev_slow_sma = slow_sma[-2]

        fan_uptrend = (prev_fast_ema > prev_medium_ema) & (prev_medium_ema > prev_slow_sma)
        fan_downtrend = (prev_fast_ema < prev_medium_ema) & (prev_medium_ema < prev_slow_sma)

        # Piercing conditions (current bar pierces through EMA)
        bull_pierce = (
            ((df["low"][-1] < prev_fast_ema) & (df["open"][-1] > prev_fast_ema) & (df["close"][-1] > prev_fast_ema)) |
            ((df["low"][-1] < prev_medium_ema) & (df["open"][-1] > prev_medium_ema) & (df["close"][-1] > prev_medium_ema)) |
            ((df["low"][-1] < prev_slow_sma) & (df["open"][-1] > prev_slow_sma) & (df["close"][-1] > prev_slow_sma))
        )

        bear_pierce = (
            ((df["high"][-1] > prev_fast_ema) & (df["open"][-1] < prev_fast_ema) & (df["close"][-1] < prev_fast_ema)) |
            ((df["high"][-1] > prev_medium_ema) & (df["open"][-1] < prev_medium_ema) & (df["close"][-1] < prev_medium_ema)) |
            ((df["high"][-1] > prev_slow_sma) & (df["open"][-1] < prev_slow_sma) & (df["close"][-1] < prev_slow_sma))
        )

        # Entry signals
        long_signal = fan_uptrend and bullish_pin_bar[-1] and bull_pierce
        short_signal = fan_downtrend and bearish_pin_bar[-1] and bear_pierce

        # EMA recross exit signals
        fast_cross_down = (prev_fast_ema >= prev_medium_ema) and (fast_ema[-1] < medium_ema[-1])
        fast_cross_up = (prev_fast_ema <= prev_medium_ema) and (fast_ema[-1] > medium_ema[-1])

        if long_signal:
            return Signal(action="buy", strength=1.0, confidence=0.8)
        elif short_signal:
            return Signal(action="sell", strength=1.0, confidence=0.8)
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "equity_risk_pct": {"type": "number", "default": 0.03, "minimum": 0.0, "maximum": 1.0},
            "atr_mult": {"type": "number", "default": 0.5, "minimum": 0.0, "maximum": 5.0},
            "atr_period": {"type": "integer", "default": 14, "minimum": 5, "maximum": 100},
            "sma_slow": {"type": "integer", "default": 50, "minimum": 20, "maximum": 200},
            "ema_medium": {"type": "integer", "default": 18, "minimum": 10, "maximum": 100},
            "ema_fast": {"type": "integer", "default": 6, "minimum": 3, "maximum": 50},
            "cancel_entry_bars": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
            "trail_points": {"type": "integer", "default": 1, "minimum": 0, "maximum": 10},
            "trail_offset": {"type": "integer", "default": 1, "minimum": 0, "maximum": 10},
            "pin_bar_ratio": {"type": "number", "default": 0.66, "minimum": 0.5, "maximum": 1.0},
        }