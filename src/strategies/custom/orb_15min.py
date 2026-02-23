from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import polars as pl
import numpy as np
import math
from datetime import time

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

def atr(high: list[float], low: list[float], close: list[float], period: int) -> list[float]:
    tr = [0.0] * len(close)
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr_values = [0.0] * len(close)
    atr_values[period - 1] = sum(tr[0:period]) / period
    for i in range(period, len(close)):
        atr_values[i] = (atr_values[i - 1] * (period - 1) + tr[i]) / period
    return atr_values

@register
class Orb15min(BaseStrategy):
    name = "orb_15min"
    version = "1.0.0"
    description = "Opening Range Breakout: 15-min ORB with scaled TP exits"
    category = "momentum"
    tags = ["breakout", "intraday"]

    def __init__(
        self,
        orb_minutes: int = 15,
        atr_period: int = 14,
        tp1_rr: float = 1.0,
        tp2_rr: float = 1.2,
        tp3_rr: float = 3.5,
        use_trailing_stop: bool = True,
        stop_buffer_pct: float = 0.001,
    ):
        self.orb_minutes = orb_minutes
        self.atr_period = atr_period
        self.tp1_rr = tp1_rr
        self.tp2_rr = tp2_rr
        self.tp3_rr = tp3_rr
        self.use_trailing_stop = use_trailing_stop
        self.stop_buffer_pct = stop_buffer_pct

    def get_parameter_schema(self) -> dict:
        return {
            "orb_minutes": {"type": "integer", "default": 15},
            "atr_period": {"type": "integer", "default": 14},
            "tp1_rr": {"type": "number", "default": 1.0},
            "tp2_rr": {"type": "number", "default": 1.2},
            "tp3_rr": {"type": "number", "default": 3.5},
            "use_trailing_stop": {"type": "boolean", "default": True},
            "stop_buffer_pct": {"type": "number", "default": 0.001},
        }

    def get_warmup_periods(self) -> int:
        return self.atr_period + 5

    def generate_signal(self, window) -> Signal | None:
        df = pl.concat([window.historical(), window.current_bar()])
        if df.height < self.get_warmup_periods():
            return None

        current_bar = df.row(df.height - 1, named=True)
        current_time = current_bar["timestamp"].time()
        trade_date = current_bar["timestamp"].date()

        orb_end_time = (time(9, 30, 0).replace(minute=9 + self.orb_minutes))

        if current_time < time(9, 30) or current_time >= time(15, 55):
            return None

        # Calculate ORB within the first orb_minutes
        orb_df = df.filter(
            (pl.col("timestamp").dt.time() >= time(9, 30)) & (pl.col("timestamp").dt.time() < orb_end_time)
        )

        if orb_df.is_empty():
            return None

        orb_high = orb_df["high"].max()
        orb_low = orb_df["low"].min()
        orb_range = orb_high - orb_low

        if orb_range <= 0:
            return None

        if current_time >= orb_end_time and current_bar["close"] > orb_high:
            # Check if this is the first breakout of the day
            past_breakouts = df.filter(
                (pl.col("timestamp").dt.date() == trade_date) &
                (pl.col("timestamp").dt.time() >= orb_end_time) &
                (pl.col("close") > orb_high)
            )
            if past_breakouts.height == 0:
                return Signal(action="buy", strength=0.8, confidence=0.8, metadata={"orb_high": orb_high, "orb_low": orb_low})

        return None