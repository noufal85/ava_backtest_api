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

def ema(series: list[float], period: int) -> list[float]:
    """Calculates the Exponential Moving Average (EMA) of a series."""
    ema = [None] * len(series)
    multiplier = 2 / (period + 1)
    
    # Initialize EMA with SMA for the first 'period' values
    ema[period - 1] = sum(series[:period]) / period
    
    # Calculate EMA for the rest of the series
    for i in range(period, len(series)):
        ema[i] = (series[i] * multiplier) + (ema[i - 1] * (1 - multiplier))
    
    # Fill initial None values with the first valid EMA value
    first_valid_ema = next((x for x in ema if x is not None), None)
    for i in range(len(series)):
        if ema[i] is None:
            ema[i] = first_valid_ema
        else:
            break
    
    return ema

@register
class TvFairValueGap(BaseStrategy):
    name = "tv_fair_value_gap"
    version = "1.0.0"
    description = "Fair Value Gap Strategy: Enters when price returns to fill gaps in trend direction"
    category = "momentum"
    tags = ["trend", "gap"]

    def __init__(self, ema_length: int = 50):
        self.ema_length = ema_length

    def get_parameter_schema(self) -> dict:
        return {
            "ema_length": {"type": "integer", "default": 50, "description": "Trend EMA length"}
        }

    def get_warmup_periods(self) -> int:
        return self.ema_length + 2 # EMA length + 2 bars for FVG lookback

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()

        if len(closes) < self.get_warmup_periods():
            return None

        ema_values = ema(closes, self.ema_length)

        # FVG Detection
        bull_fvg = lows[-1] > highs[-3] if len(highs) >= 3 else False
        bear_fvg = highs[-1] < lows[-3] if len(lows) >= 3 else False

        bull_fvg_top = lows[-1] if bull_fvg else None
        bull_fvg_bot = highs[-3] if bull_fvg else None
        bear_fvg_top = lows[-3] if bear_fvg else None
        bear_fvg_bot = highs[-1] if bear_fvg else None

        # Active FVG tracking (simplified - single bar lookback)
        prev_row = historical_data.tail(1)
        prev_ema = ema_values[-2]
        
        prev_bull_fvg_top = prev_row["active_bull_fvg_top"][0] if "active_bull_fvg_top" in prev_row.columns else None
        prev_bull_fvg_bot = prev_row["active_bull_fvg_bot"][0] if "active_bull_fvg_bot" in prev_row.columns else None
        prev_bear_fvg_top = prev_row["active_bear_fvg_top"][0] if "active_bear_fvg_top" in prev_row.columns else None
        prev_bear_fvg_bot = prev_row["active_bear_fvg_bot"][0] if "active_bear_fvg_bot" in prev_row.columns else None

        active_bull_fvg_top = bull_fvg_top if bull_fvg else prev_bull_fvg_top
        active_bull_fvg_bot = bull_fvg_bot if bull_fvg else prev_bull_fvg_bot
        active_bear_fvg_top = bear_fvg_top if bear_fvg else prev_bear_fvg_top
        active_bear_fvg_bot = bear_fvg_bot if bear_fvg else prev_bear_fvg_bot

        # Clear FVG when filled
        if active_bull_fvg_bot is not None and lows[-1] <= active_bull_fvg_bot:
            active_bull_fvg_top = None
            active_bull_fvg_bot = None
        if active_bear_fvg_top is not None and highs[-1] >= active_bear_fvg_top:
            active_bear_fvg_top = None
            active_bear_fvg_bot = None

        # Signal generation
        long_condition = (
            closes[-1] > prev_ema and
            active_bull_fvg_top is not None and
            lows[-1] <= active_bull_fvg_top and
            lows[-1] >= active_bull_fvg_bot
        )

        short_condition = (
            closes[-1] < prev_ema and
            active_bear_fvg_bot is not None and
            highs[-1] >= active_bear_fvg_bot and
            highs[-1] <= active_bear_fvg_top
        )

        if long_condition:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"fvg_type": "bullish"})
        #if short_condition:
        #    return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"fvg_type": "bearish"})

        return None