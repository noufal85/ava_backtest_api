"""bb_squeeze — converted from trading-backtester."""
import polars as pl
import numpy as np
import math
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register

@register
class BbSqueeze(BaseStrategy):
    name = "bb_squeeze"
    name: str = "bb_squeeze"
    version: str = "1.1.0"
    description: str = "Bollinger Squeeze: breakout after contraction (long & short)"
    category: str = "mean_reversion"
    tags: list[str] = ["bollinger bands", "squeeze", "breakout"]

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, squeeze_lookback: int = 120, squeeze_pctile: float = 10.0):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_pctile = squeeze_pctile

    def get_parameter_schema(self) -> dict:
        return {
            "bb_period": {"type": "integer", "default": 20},
            "bb_std": {"type": "number", "default": 2.0},
            "squeeze_lookback": {"type": "integer", "default": 120},
            "squeeze_pctile": {"type": "number", "default": 10.0},
        }

    def get_warmup_periods(self) -> int:
        return self.squeeze_lookback + 5  # Add a small buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if df.height < self.get_warmup_periods():
            return None

        close_prices = df["close"]
        
        bb = bollinger_bands(close_prices, period=self.bb_period, std=self.bb_std)
        upper_band = bb["upper"]
        middle_band = bb["middle"]
        lower_band = bb["lower"]
        bandwidth = bb["bandwidth"]

        if upper_band is None or middle_band is None or lower_band is None or bandwidth is None:
            return None

        # Calculate bandwidth percentile
        if df.height < self.squeeze_lookback:
            return None

        bandwidth_values = bandwidth.tail(self.squeeze_lookback).drop_nulls().to_list()
        if len(bandwidth_values) < self.squeeze_lookback:
            return None

        current_bandwidth = bandwidth_values[-1]
        percentile = sum(1 for x in bandwidth_values if x <= current_bandwidth) / len(bandwidth_values) * 100

        in_squeeze = percentile <= self.squeeze_pctile

        # Get previous values
        prev_close = close_prices[-2]
        prev_upper = upper_band[-2]
        prev_lower = lower_band[-2]

        # Long breakout: price above upper band
        breakout_up = prev_close > prev_upper

        # Short breakout: price breaks above upper band WITHOUT squeeze (overbought)
        breakout_down = prev_close > prev_upper # SHORT when price breaks ABOVE upper band

        if in_squeeze and breakout_up:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"bandwidth": current_bandwidth})
        elif (prev_close > prev_upper) and not in_squeeze:
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"bandwidth": current_bandwidth})
        else:
            return None