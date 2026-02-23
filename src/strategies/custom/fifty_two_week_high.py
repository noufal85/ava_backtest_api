"""fifty_two_week_high — converted from trading-backtester."""
import polars as pl
import numpy as np
import math
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register

@register
class FiftyTwoWeekHigh(BaseStrategy):
    name: str = "fifty_two_week_high"
    version: str = "1.0.0"
    description: str = "Momentum: buy near 52-week high with positive yearly return"
    category: str = "multi_factor"
    tags: list[str] = ["momentum", "52-week high"]

    def __init__(self, lookback: int = 252, proximity_pct: float = 0.05, hold_days: int = 20, stop_pct: float = 0.1):
        self.lookback = lookback
        self.proximity_pct = proximity_pct
        self.hold_days = hold_days
        self.stop_pct = stop_pct

    def get_parameter_schema(self) -> dict:
        return {
            "lookback": {"type": "integer", "default": 252},
            "proximity_pct": {"type": "number", "default": 0.05},
            "hold_days": {"type": "integer", "default": 20},
            "stop_pct": {"type": "number", "default": 0.1},
        }

    def get_warmup_periods(self) -> int:
        return self.lookback + 10 # Buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.lookback:
            return None

        # Calculate indicators
        high_val = n_day_high(df, self.lookback).item(-2)
        ret_val = n_day_return(df, self.lookback).item(-2)
        low_val = df["close"].rolling(window_size=self.lookback, min_periods=self.lookback).min().item(-2)
        ten_d_low = df["close"].rolling(window_size=10, min_periods=10).min().item(-2)
        five_d_ret = (df["close"].item(-2) / df["close"].shift(5).item(-2) - 1) if len(df) >= 6 else 0
        new_high = df["close"].item(-2) > df["high"].rolling(window_size=self.lookback, min_periods=self.lookback).max().shift(1).item(-1) if len(df) > self.lookback else False

        close_prev = df["close"].item(-2)

        # Entry conditions
        proximity = (high_val - close_prev) / high_val if high_val != 0 else 0
        buy_mask = (proximity <= self.proximity_pct) and (ret_val > 0)

        # Short: near 52-week low (within 5%) and negative yearly return
        low_proximity = (close_prev - low_val) / low_val if low_val != 0 else 0
        short_mask = (low_proximity <= self.proximity_pct) and (ret_val < 0)

        # Exit conditions
        momentum_breakdown = close_prev < ten_d_low
        profit_take = new_high
        momentum_exhaustion = five_d_ret < -0.05

        exit_mask = (momentum_breakdown or profit_take or momentum_exhaustion) and not (buy_mask or short_mask)

        if buy_mask:
            return Signal(action="buy", strength=float(ret_val), confidence=1.0, metadata={"proximity": proximity})
        elif short_mask:
            return Signal(action="sell", strength=float(abs(ret_val)), confidence=1.0, metadata={"proximity": low_proximity})
        elif exit_mask:
            return Signal(action="sell", strength=1.0, confidence=1.0)
        else:
            return None