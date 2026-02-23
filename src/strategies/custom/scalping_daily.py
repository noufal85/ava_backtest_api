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

def ema(series: pl.Series, period: int) -> pl.Series:
    alpha = 2 / (period + 1)
    ema = [series[0]]
    for i in range(1, len(series)):
        ema.append(alpha * series[i] + (1 - alpha) * ema[-1])
    return pl.Series(ema)

def rsi(series: pl.Series, period: int) -> pl.Series:
    delta = series.diff().drop_nulls()
    up, down = delta.clone(), delta.clone()
    up = up.with_columns(pl.when(up < 0).then(0).otherwise(up))
    down = down.with_columns(pl.when(down > 0).then(0).otherwise(down.abs()))

    avg_gain = up.rolling(period).mean().extend_null(period - 1)
    avg_loss = down.rolling(period).mean().extend_null(period - 1)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def atr(df: pl.DataFrame, period: int) -> pl.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1).fill_null(close[0]))
    tr3 = abs(low - close.shift(1).fill_null(close[0]))
    tr = pl.max([tr1, tr2, tr3])
    atr = tr.rolling(period).mean().extend_null(period - 1)
    return atr

def bollinger_bands(series: pl.Series, period: int, std: float) -> dict:
    middle = series.rolling(period).mean().extend_null(period - 1)
    std_dev = series.rolling(period).std().extend_null(period - 1)
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return {"upper": upper, "middle": middle, "lower": lower}

@register
class ScalpingDaily(BaseStrategy):
    name = "scalping_daily"
    version = "1.0.0"
    description = "Scalping adapted for daily bars: quick entries/exits on small moves"
    category = "multi_factor"
    tags = ["scalping", "daily", "multi-factor"]

    def __init__(
        self,
        ema_fast: int = 5,
        ema_slow: int = 13,
        rsi_period: int = 7,
        rsi_entry: float = 40.0,
        rsi_exit: float = 65.0,
        atr_period: int = 14,
        atr_stop_mult: float = 1.0,
        max_hold: int = 3,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_entry = rsi_entry
        self.rsi_exit = rsi_exit
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.max_hold = max_hold

    def get_warmup_periods(self) -> int:
        return max(self.ema_fast, self.ema_slow, self.rsi_period, self.atr_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.get_warmup_periods():
            return None

        closes = df["close"]
        highs = df["high"]
        lows = df["low"]

        ema_fast = ema(closes, self.ema_fast)
        ema_slow = ema(closes, self.ema_slow)
        rsi_values = rsi(closes, self.rsi_period)
        atr_values = atr(df, self.atr_period)
        bb = bollinger_bands(closes, 20, 2.0)
        bb_lower = bb["lower"]

        current_rsi = rsi_values[-1]
        current_ema_fast = ema_fast[-1]
        current_ema_slow = ema_slow[-1]
        current_close = closes[-1]
        current_bb_lower = bb_lower[-1]

        previous_rsi = rsi_values[-2]
        previous_ema_fast = ema_fast[-2]
        previous_ema_slow = ema_slow[-2]
        previous_close = closes[-2]
        previous_bb_lower = bb_lower[-2]

        oversold = previous_rsi < self.rsi_entry
        near_bb_lower = previous_close <= previous_bb_lower * 1.02
        uptrend = previous_ema_fast > previous_ema_slow

        if oversold and near_bb_lower and uptrend:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"strategy": "scalping_daily"})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "ema_fast": {"type": "integer", "default": 5, "description": "Fast EMA period"},
            "ema_slow": {"type": "integer", "default": 13, "description": "Slow EMA period"},
            "rsi_period": {"type": "integer", "default": 7, "description": "RSI period"},
            "rsi_entry": {"type": "number", "default": 40.0, "description": "RSI oversold level to enter"},
            "rsi_exit": {"type": "number", "default": 65.0, "description": "RSI overbought level to exit"},
            "atr_period": {"type": "integer", "default": 14, "description": "ATR period"},
            "atr_stop_mult": {"type": "number", "default": 1.0, "description": "ATR multiplier for tight stop"},
            "max_hold": {"type": "integer", "default": 3, "description": "Max holding days"},
        }