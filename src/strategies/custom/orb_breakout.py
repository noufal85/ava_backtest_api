from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import time as dt_time
import math

import numpy as np
import polars as pl

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
    """Compute Average True Range (ATR) using Polars."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr0 = high - low
    tr1 = abs(high - close.shift(1).fill_null(0.0))
    tr2 = abs(low - close.shift(1).fill_null(0.0))

    tr = pl.max([tr0, tr1, tr2])

    atr_values = tr.rolling_mean(window_size=period, min_periods=1)
    return atr_values.alias("atr")


@register
class OrbBreakout(BaseStrategy):
    name = "orb_breakout"
    version = "1.0.0"
    description = "Opening Range Breakout: intraday breakout with volume confirmation"
    category = "momentum"
    tags = ["breakout", "intraday", "volume confirmation"]

    def __init__(
        self,
        or_minutes: int = 30,
        or_bars: int = 2,
        volume_mult: float = 3.0,
        volume_lookback: int = 10,
        atr_period: int = 10,
        stop_atr_mult: float = 1.5,
        reward_risk_ratio: float = 2.0,
        max_hold_bars: int = 26,
        allow_long: bool = True,
        allow_short: bool = False,
        daily_proxy: bool = False,
        gap_threshold: float = 0.005,
    ):
        self.or_minutes = or_minutes
        self.or_bars = or_bars
        self.volume_mult = volume_mult
        self.volume_lookback = volume_lookback
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        self.reward_risk_ratio = reward_risk_ratio
        self.max_hold_bars = max_hold_bars
        self.allow_long = allow_long
        self.allow_short = allow_short
        self.daily_proxy = daily_proxy
        self.gap_threshold = gap_threshold

    def get_parameter_schema(self) -> dict:
        return {
            "or_minutes": {"type": "integer", "default": 30},
            "or_bars": {"type": "integer", "default": 2},
            "volume_mult": {"type": "number", "default": 3.0},
            "volume_lookback": {"type": "integer", "default": 10},
            "atr_period": {"type": "integer", "default": 10},
            "stop_atr_mult": {"type": "number", "default": 1.5},
            "reward_risk_ratio": {"type": "number", "default": 2.0},
            "max_hold_bars": {"type": "integer", "default": 26},
            "allow_long": {"type": "boolean", "default": True},
            "allow_short": {"type": "boolean", "default": False},
            "daily_proxy": {"type": "boolean", "default": False},
            "gap_threshold": {"type": "number", "default": 0.005},
        }

    def get_warmup_periods(self) -> int:
        return self.volume_lookback + self.atr_period + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if self.daily_proxy:
            return self._generate_daily_signal(df)
        else:
            return self._generate_intraday_signal(df)

    def _generate_intraday_signal(self, df: pl.DataFrame) -> Signal | None:
        if len(df) < self.get_warmup_periods():
            return None

        # Compute opening range (OR) high and low
        or_high = df.head(self.or_bars)["high"].max()
        or_low = df.head(self.or_bars)["low"].min()

        # Get current bar data
        current_close = df["close"][-1]
        current_volume = df["volume"][-1]

        # Calculate average daily volume
        daily_volume = df.group_by(pl.col("timestamp").dt.date()).agg(pl.col("volume").sum())
        avg_daily_volume = daily_volume["volume"].tail(self.volume_lookback).mean()

        # Calculate ATR
        atr_series = atr(df, self.atr_period)
        current_atr = atr_series[-1]

        # Volume confirmation
        avg_bar_volume = avg_daily_volume / 26  # Approximate bars per day
        vol_confirmed = current_volume > (self.volume_mult * avg_bar_volume)

        # Breakout conditions
        breakout_up = current_close > or_high and vol_confirmed
        breakout_down = current_close < or_low and vol_confirmed

        if self.allow_long and breakout_up:
            strength = min(3.0, (current_close - or_high) / (or_high - or_low)) if (or_high - or_low) != 0 else 0.0
            return Signal(action="buy", strength=strength, confidence=1.0)

        if self.allow_short and breakout_down:
            strength = min(3.0, (or_low - current_close) / (or_high - or_low)) if (or_high - or_low) != 0 else 0.0
            return Signal(action="sell", strength=strength, confidence=1.0)

        return None

    def _generate_daily_signal(self, df: pl.DataFrame) -> Signal | None:
        if len(df) < self.get_warmup_periods():
            return None

        # Shifted values
        prev_high = df["high"][-2]
        prev_low = df["low"][-2]
        prev_close = df["close"][-2]

        # Current values
        current_open = df["open"][-1]
        current_volume = df["volume"][-1]

        # Calculate average volume
        avg_volume = df["volume"].tail(self.volume_lookback).mean()

        # Calculate ATR
        atr_series = atr(df, self.atr_period)
        current_atr = atr_series[-1]

        # Volume confirmation
        vol_ok = current_volume > (self.volume_mult * avg_volume)

        # Long condition: open gaps above previous high
        gap_up = current_open > prev_high * (1 + self.gap_threshold) if not math.isnan(prev_high) else False
        if self.allow_long and gap_up and vol_ok:
            strength = min(0.1, (current_open - prev_high) / prev_close) if not math.isnan(prev_close) else 0.0
            return Signal(action="buy", strength=strength, confidence=1.0)

        # Short condition: open gaps below previous low
        gap_down = current_open < prev_low * (1 - self.gap_threshold) if not math.isnan(prev_low) else False
        if self.allow_short and gap_down and vol_ok:
            strength = min(0.1, (prev_low - current_open) / prev_close) if not math.isnan(prev_close) else 0.0
            return Signal(action="sell", strength=strength, confidence=1.0)

        return None