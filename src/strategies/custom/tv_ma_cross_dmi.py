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

def ema(series: pl.Series, period: int) -> pl.Series:
    """Calculates the Exponential Moving Average (EMA) of a series."""
    alpha = 2 / (period + 1)
    ema = [series[0]]
    for i in range(1, len(series)):
        ema.append(alpha * series[i] + (1 - alpha) * ema[-1])
    return pl.Series(ema)

def sma(series: pl.Series, period: int) -> pl.Series:
    """Calculates the Simple Moving Average (SMA) of a series."""
    if len(series) < period:
        return pl.Series([None] * len(series))
    sma_values = []
    for i in range(len(series)):
        if i < period - 1:
            sma_values.append(None)
        else:
            sma_values.append(series[i-period+1:i+1].mean())
    return pl.Series(sma_values)

def true_range(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """Calculates the True Range (TR) for each period."""
    tr = []
    for i in range(len(high)):
        if i == 0:
            tr.append(high[i] - low[i])
        else:
            tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    return pl.Series(tr)

def directional_movement(high: pl.Series, low: pl.Series) -> tuple[pl.Series, pl.Series]:
    """Calculates the +DM and -DM for each period."""
    plus_dm = []
    minus_dm = []
    for i in range(len(high)):
        if i == 0:
            plus_dm.append(0.0)
            minus_dm.append(0.0)
        else:
            move_up = high[i] - high[i-1]
            move_down = low[i-1] - low[i]
            if move_up > move_down and move_up > 0:
                plus_dm.append(move_up)
            else:
                plus_dm.append(0.0)
            if move_down > move_up and move_down > 0:
                minus_dm.append(move_down)
            else:
                minus_dm.append(0.0)
    return pl.Series(plus_dm), pl.Series(minus_dm)

def directional_indicators(df: pl.DataFrame, period: int) -> tuple[pl.Series, pl.Series]:
    """Calculates the +DI and -DI."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr = true_range(high, low, close)
    plus_dm, minus_dm = directional_movement(high, low)
    
    atr = ema(tr, period)
    plus_di = 100 * (ema(plus_dm, period) / atr)
    minus_di = 100 * (ema(minus_dm, period) / atr)
    
    return plus_di, minus_di

def adx(df: pl.DataFrame, period: int, smoothing: int) -> pl.Series:
    """Calculates the Average Directional Index (ADX)."""
    plus_di, minus_di = directional_indicators(df, period)
    
    di_sum = plus_di + minus_di
    di_diff = abs(plus_di - minus_di)
    
    dx = pl.Series([0.0 if s == 0 else 100 * (d / s) for d, s in zip(di_diff, di_sum)])
    adx_values = ema(dx, smoothing)
    return adx_values

@register
class TvMaCrossDmi(BaseStrategy):
    name = "tv_ma_cross_dmi"
    version = "1.0.0"
    description = "MA Cross + DMI: Moving average crossover with directional movement filter"
    category = "trend"
    tags = ["ma", "crossover", "dmi", "trend"]

    def __init__(
        self,
        ma1_type: str = "EMA",
        ma1_length: int = 10,
        ma2_type: str = "EMA",
        ma2_length: int = 20,
        dmi_length: int = 14,
        adx_smoothing: int = 13,
        key_level: float = 23.0,
        use_stop_loss: bool = False,
        stop_loss_pct: float = 10.0,
    ):
        self.ma1_type = ma1_type
        self.ma1_length = ma1_length
        self.ma2_type = ma2_type
        self.ma2_length = ma2_length
        self.dmi_length = dmi_length
        self.adx_smoothing = adx_smoothing
        self.key_level = key_level
        self.use_stop_loss = use_stop_loss
        self.stop_loss_pct = stop_loss_pct

    def get_parameter_schema(self) -> dict:
        return {
            "ma1_type": {"type": "string", "enum": ["EMA", "SMA"], "default": self.ma1_type},
            "ma1_length": {"type": "integer", "minimum": 1, "default": self.ma1_length},
            "ma2_type": {"type": "string", "enum": ["EMA", "SMA"], "default": self.ma2_type},
            "ma2_length": {"type": "integer", "minimum": 1, "default": self.ma2_length},
            "dmi_length": {"type": "integer", "minimum": 1, "default": self.dmi_length},
            "adx_smoothing": {"type": "integer", "minimum": 1, "default": self.adx_smoothing},
            "key_level": {"type": "number", "default": self.key_level},
            "use_stop_loss": {"type": "boolean", "default": self.use_stop_loss},
            "stop_loss_pct": {"type": "number", "default": self.stop_loss_pct},
        }

    def get_warmup_periods(self) -> int:
        return max(self.ma1_length, self.ma2_length, self.dmi_length, self.adx_smoothing) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.get_warmup_periods():
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Calculate moving averages
        if self.ma1_type == "EMA":
            ma1 = ema(close, self.ma1_length)
        else:
            ma1 = sma(close, self.ma1_length)

        if self.ma2_type == "EMA":
            ma2 = ema(close, self.ma2_length)
        else:
            ma2 = sma(close, self.ma2_length)

        # Calculate DMI indicators
        di_plus, di_minus = directional_indicators(df, self.dmi_length)
        adx_values = adx(df, self.dmi_length, self.adx_smoothing)

        # MA crossover detection (using last 3 values to avoid lookahead)
        if len(ma1) < 3 or len(ma2) < 3:
            return None

        prev_ma1 = ma1[-2]
        prev_ma2 = ma2[-2]
        prev2_ma1 = ma1[-3]
        prev2_ma2 = ma2[-3]

        ma_bullish_cross = (prev_ma1 > prev_ma2) and (prev2_ma1 <= prev2_ma2)
        ma_bearish_cross = (prev_ma1 < prev_ma2) and (prev2_ma1 >= prev2_ma2)

        # DMI conditions (as defined in original but not actively used)
        prev_di_plus = di_plus[-2]
        prev_di_minus = di_minus[-2]
        dmi_long_cond = (prev_di_plus < prev_di_minus)
        dmi_short_cond = (prev_di_plus > prev_di_minus)

        # Entry signals (currently just based on MA crossovers as in original)
        long_entry = ma_bullish_cross
        short_entry = ma_bearish_cross  # Disabled for long-only

        current_price = close[-1]

        if long_entry:
            return Signal(
                action="buy",
                strength=1.0,
                confidence=1.0,
                metadata={
                    "ma1": ma1[-1],
                    "ma2": ma2[-1],
                    "di_plus": di_plus[-1],
                    "di_minus": di_minus[-1],
                    "adx": adx_values[-1],
                },
            )

        return None