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

def sma(series: pl.Series, period: int) -> pl.Series:
    """Simple Moving Average."""
    if len(series) < period:
        return pl.Series([None] * len(series))
    return series.rolling_mean(window_size=period, min_periods=period, center=False)

def bollinger_bands(series: pl.Series, period: int, std_dev: float) -> tuple[pl.Series, pl.Series, pl.Series]:
    """Bollinger Bands."""
    middle_band = sma(series, period)
    std = series.rolling_std(window_size=period, min_periods=period, center=False)
    upper_band = middle_band + std_dev * std
    lower_band = middle_band - std_dev * std
    return upper_band, lower_band, middle_band

@register
class TvWilliamsVixFix(BaseStrategy):
    name = "tv_williams_vix_fix"
    version = "1.0.0"
    description = "Williams VIX Fix oscillator for mean reversion trading"
    category = "volatility"
    tags = ["mean_reversion", "volatility"]

    def __init__(self, bb_length: int = 20, bb_mult: float = 2.0, wvf_pd: int = 20, wvf_lb: int = 50, wvf_ph: float = 85.0, wvf_pl: float = 99.0):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.wvf_pd = wvf_pd
        self.wvf_lb = wvf_lb
        self.wvf_ph = wvf_ph / 100.0
        self.wvf_pl = wvf_pl / 100.0

    def get_warmup_periods(self) -> int:
        return max(self.bb_length, self.wvf_pd, self.wvf_lb) + 2

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.get_warmup_periods():
            return None

        # Williams VIX Fix
        highest_close = df["close"].rolling_max(self.wvf_pd).to_numpy()
        wvf = ((highest_close - df["low"].to_numpy()) / highest_close) * 100

        # WVF bands and thresholds
        wvf_series = pl.Series(wvf)
        wvf_std = wvf_series.rolling_std(window_size=self.bb_length).to_numpy()
        wvf_midline = sma(wvf_series, period=self.bb_length).to_numpy()
        wvf_upper_band = wvf_midline + self.bb_mult * wvf_std
        wvf_range_high = pl.Series(wvf).rolling_max(window_size=self.wvf_lb).to_numpy() * self.wvf_ph

        # Williams VIX Fix Inverted
        lowest_close = df["close"].rolling_min(self.wvf_pd).to_numpy()
        wvf_inv = ((df["high"].to_numpy() - lowest_close) / lowest_close) * 100

        # WVF Inverted bands and thresholds
        wvf_inv_series = pl.Series(wvf_inv)
        wvf_inv_std = wvf_inv_series.rolling_std(window_size=self.bb_length).to_numpy()
        wvf_inv_midline = sma(wvf_inv_series, period=self.bb_length).to_numpy()
        wvf_inv_upper_band = wvf_inv_midline + self.bb_mult * wvf_inv_std
        wvf_inv_range_high = pl.Series(wvf_inv).rolling_min(window_size=self.wvf_lb).to_numpy() * self.wvf_pl

        # Bollinger Bands
        upper_band, lower_band, _ = bollinger_bands(df["close"], self.bb_length, self.bb_mult)
        upper_band = upper_band.to_numpy()
        lower_band = lower_band.to_numpy()

        # Use shifted indicators to avoid look-ahead bias
        prev_wvf = wvf[-2]
        prev_wvf_upper_band = wvf_upper_band[-2]
        prev_wvf_range_high = wvf_range_high[-2]
        prev_close = df["close"][-2]
        prev_bb_lower = lower_band[-2]
        prev_bb_upper = upper_band[-2]

        prev_wvf_inv = wvf_inv[-2]
        prev_wvf_inv_upper_band = wvf_inv_upper_band[-2]
        prev_wvf_inv_range_high = wvf_inv_range_high[-2]

        # WVF signals
        wvf_signal = (
            (prev_wvf >= prev_wvf_upper_band) or 
            (prev_wvf >= prev_wvf_range_high)
        ) and (
            (wvf[-3] >= wvf_upper_band[-3]) or
            (wvf[-3] >= wvf_range_high[-3])
        )

        wvf_inv_signal = (
            (prev_wvf_inv <= prev_wvf_inv_upper_band) or
            (prev_wvf_inv <= prev_wvf_inv_range_high)
        ) and (
            (wvf_inv[-3] <= wvf_inv_upper_band[-3]) or
            (wvf_inv[-3] <= wvf_inv_range_high[-3])
        )

        # Entry and exit signals
        long_signal = wvf_signal and (prev_close < prev_bb_lower)
        long_exit = wvf_inv_signal and (df["close"][-1] > prev_bb_upper)

        if long_signal:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"wvf": wvf[-1], "bb_lower": lower_band[-1]})
        elif long_exit:
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"wvf_inv": wvf_inv[-1], "bb_upper": upper_band[-1]})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "bb_length": {"type": "integer", "default": 20, "title": "BB Length"},
            "bb_mult": {"type": "number", "default": 2.0, "title": "BB Mult"},
            "wvf_pd": {"type": "integer", "default": 20, "title": "WVF Pd"},
            "wvf_lb": {"type": "integer", "default": 50, "title": "WVF Lb"},
            "wvf_ph": {"type": "number", "default": 85.0, "title": "WVF Ph"},
            "wvf_pl": {"type": "number", "default": 99.0, "title": "WVF Pl"},
        }