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

def atr(df: pl.DataFrame, period: int) -> pl.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1).fill_null(0.0))
    tr3 = abs(low - close.shift(1).fill_null(0.0))
    tr = pl.max([tr1, tr2, tr3])

    atr_list = []
    for i in range(len(df)):
        if i < period:
            atr_list.append(np.nan)
        else:
            atr_list.append(tr[i-period+1:i+1].mean())
    return pl.Series(atr_list)

def ema(df: pl.DataFrame, period: int) -> pl.Series:
    """Exponential Moving Average."""
    close = df["close"]
    ema_list = []
    alpha = 2 / (period + 1)
    
    for i in range(len(df)):
        if i < period-1:
            ema_list.append(np.nan)
        elif i == period-1:
            ema_list.append(close[:period].mean())
        else:
            ema_list.append(alpha * close[i] + (1 - alpha) * ema_list[-1])
    return pl.Series(ema_list)

@register
class KellBaseBreakout(BaseStrategy):
    name = "kell_base_breakout"
    version = "2.0.0"
    description = "Oliver Kell base-and-breakout: buy breakouts from tight consolidation bases"
    category = "momentum"
    tags = ["breakout", "momentum", "kell"]

    def __init__(
        self,
        atr_period: int = 14,
        base_period: int = 20,
        base_atr_mult: float = 4.0,
        ema_fast: int = 10,
        ema_slow: int = 20,
        vol_mult: float = 1.3,
        risk_pct: float = 0.02,
    ):
        self.atr_period = atr_period
        self.base_period = base_period
        self.base_atr_mult = base_atr_mult
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.vol_mult = vol_mult
        self.risk_pct = risk_pct

    def get_parameter_schema(self) -> dict:
        return {
            "atr_period": {"type": "integer", "default": 14},
            "base_period": {"type": "integer", "default": 20},
            "base_atr_mult": {"type": "number", "default": 4.0},
            "ema_fast": {"type": "integer", "default": 10},
            "ema_slow": {"type": "integer", "default": 20},
            "vol_mult": {"type": "number", "default": 1.3},
            "risk_pct": {"type": "number", "default": 0.02},
        }

    def get_warmup_periods(self) -> int:
        return max(self.atr_period, self.base_period, self.ema_fast, self.ema_slow) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_df = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_df, current_bar])

        if len(df) < self.get_warmup_periods():
            return None

        # Indicator calculation
        df = df.with_columns(atr=atr(df, self.atr_period))
        df = df.with_columns(ema_fast=ema(df, self.ema_fast))
        df = df.with_columns(ema_slow=ema(df, self.ema_slow))

        base_high = df["close"].shift(1).rolling(window=self.base_period, min_periods=self.base_period).max().alias("base_high")
        base_low = df["close"].shift(1).rolling(window=self.base_period, min_periods=self.base_period).min().alias("base_low")
        df = df.with_columns(base_high=base_high)
        df = df.with_columns(base_low=base_low)
        df = df.with_columns(base_range=(pl.col("base_high") - pl.col("base_low")))
        df = df.with_columns(avg_volume=df["volume"].rolling(window=self.base_period, min_periods=self.base_period).mean())

        # Entry conditions
        current_row = df.row(len(df) - 1, named=True)
        prev_row = df.row(len(df) - 2, named=True)

        in_base = current_row["base_range"] < (self.base_atr_mult * prev_row["atr"])
        uptrend = prev_row["ema_fast"] > prev_row["ema_slow"]
        breakout = current_row["close"] > current_row["base_high"]
        volume_confirm = current_row["volume"] > (self.vol_mult * current_row["avg_volume"])

        # Exit conditions - EMA cross down (use previous bar to avoid look-ahead)
        ema_fast_now = current_row["ema_fast"]
        ema_slow_now = current_row["ema_slow"]
        ema_fast_prev = prev_row["ema_fast"]
        ema_slow_prev = prev_row["ema_slow"]

        ema_cross_down = (ema_fast_prev >= ema_slow_prev) and (ema_fast_now < ema_slow_now)

        if in_base and uptrend and breakout and volume_confirm:
            return Signal(action="buy", strength=1.0, confidence=1.0)
        elif ema_cross_down:
            return Signal(action="sell", strength=1.0, confidence=1.0)
        else:
            return None