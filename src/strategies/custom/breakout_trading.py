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
    """Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1).fill_null(0))
    tr3 = abs(low - close.shift(1).fill_null(0))

    tr = pl.max([tr1, tr2, tr3])
    atr_values = tr.rolling_mean(window=period, min_periods=period)
    return atr_values.alias("atr")

def ema(df: pl.DataFrame, period: int) -> pl.Series:
    """Exponential Moving Average (EMA)."""
    close = df["close"]
    alpha = 2 / (period + 1)
    ema_values = close.ewm_mean(alpha=alpha, adjust=False)
    return ema_values.alias("ema")

def volume_sma(df: pl.DataFrame, period: int) -> pl.Series:
    """Volume Simple Moving Average (SMA)."""
    volume = df["volume"]
    sma_values = volume.rolling_mean(window=period, min_periods=period)
    return sma_values.alias("vol_sma")

@register
class BreakoutTrading(BaseStrategy):
    name = "breakout_trading"
    version = "1.0.0"
    description = "Breakout trading: enter on resistance breakout with volume confirmation"
    category = "momentum"
    tags = ["breakout", "volume", "trend"]

    def __init__(
        self,
        lookback: int = 20,
        vol_mult: float = 1.5,
        vol_sma_period: int = 20,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        ema_period: int = 50,
        max_hold: int = 15,
    ):
        self.lookback = lookback
        self.vol_mult = vol_mult
        self.vol_sma_period = vol_sma_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.ema_period = ema_period
        self.max_hold = max_hold

    def get_warmup_periods(self) -> int:
        return max(self.lookback, self.vol_sma_period, self.atr_period, self.ema_period) + 5

    def generate_signal(self, window) -> Signal | None:
        df = pl.concat([window.historical(), window.current_bar()])
        if len(df) < self.get_warmup_periods():
            return None

        # Indicator Calculation
        df = df.with_columns([
            atr(df, self.atr_period),
            ema(df, self.ema_period),
            volume_sma(df, self.vol_sma_period)
        ])

        high = df["high"].to_list()
        low = df["low"].to_list()
        close = df["close"].to_list()
        volume = df["volume"].to_list()

        resistance = [None] * len(df)
        support = [None] * len(df)

        for i in range(1, len(df)):
            if i >= self.lookback:
                resistance[i] = max(high[i-self.lookback:i])
                support[i] = min(low[i-self.lookback:i])

        df = df.with_columns([
            pl.Series(name="resistance", values=resistance),
            pl.Series(name="support", values=support)
        ])

        # Signal Logic
        current_row = df.row(len(df) - 1, named=True)
        prev_row = df.row(len(df) - 2, named=True) if len(df) > 1 else None

        if prev_row is None or current_row["resistance"] is None:
            return None

        prev_close = prev_row["close"]
        prev_ema = prev_row["ema"]
        prev_volume = prev_row["volume"]
        prev_vol_sma = prev_row["vol_sma"]
        prev_resistance = prev_row["resistance"]

        breakout = prev_close > prev_resistance
        volume_confirm = prev_volume > (self.vol_mult * prev_vol_sma)
        uptrend = prev_close > prev_ema

        if breakout and volume_confirm and uptrend:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"strategy": "breakout_trading"})

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "lookback": {"type": "integer", "default": 20},
                "vol_mult": {"type": "number", "default": 1.5},
                "vol_sma_period": {"type": "integer", "default": 20},
                "atr_period": {"type": "integer", "default": 14},
                "atr_stop_mult": {"type": "number", "default": 2.0},
                "ema_period": {"type": "integer", "default": 50},
                "max_hold": {"type": "integer", "default": 15},
            },
        }