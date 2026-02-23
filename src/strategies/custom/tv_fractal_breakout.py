from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

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

@register
class TvFractalBreakoutStrategy(BaseStrategy):
    name = "tv_fractal_breakout"
    version = "1.0.0"
    description = "Williams Fractal Breakout: Long on fractal high breakouts"
    category = "momentum"
    tags = ["fractal", "breakout"]

    def __init__(
        self,
        fractal_period: int = 2,
        exit_on_fractal_low: bool = True,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ):
        self.fractal_period = fractal_period
        self.exit_on_fractal_low = exit_on_fractal_low
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def get_parameter_schema(self) -> dict:
        return {
            "fractal_period": {"type": "integer", "default": 2, "minimum": 1},
            "exit_on_fractal_low": {"type": "boolean", "default": True},
            "stop_loss_pct": {"type": ["number", "null"], "default": None, "minimum": 0.0, "maximum": 1.0},
            "take_profit_pct": {"type": ["number", "null"], "default": None, "minimum": 0.0, "maximum": 1.0},
        }

    def get_warmup_periods(self) -> int:
        return 2 * self.fractal_period + 5  # Fractal period + small buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        n = self.fractal_period
        if len(df) < 2 * n + 1:
            return None

        highs = df["high"].to_list()
        lows = df["low"].to_list()
        closes = df["close"].to_list()

        # Fractal High/Low Calculation
        fractal_highs = [False] * len(df)
        fractal_lows = [False] * len(df)

        for i in range(n, len(df) - n):
            window_highs = highs[i - n : i + n + 1]
            if highs[i] == max(window_highs):
                fractal_highs[i] = True

            window_lows = lows[i - n : i + n + 1]
            if lows[i] == min(window_lows):
                fractal_lows[i] = True

        # Last Fractal High/Low Tracking
        last_fractal_high = None
        last_fractal_low = None
        last_fractal_highs = [None] * len(df)
        last_fractal_lows = [None] * len(df)

        for i in range(len(df)):
            if fractal_highs[i]:
                last_fractal_high = highs[i]
            if fractal_lows[i]:
                last_fractal_low = lows[i]

            last_fractal_highs[i] = last_fractal_high
            last_fractal_lows[i] = last_fractal_low

        # Signal Generation (using the *previous* values to avoid lookahead)
        if len(df) >= 2:
            prev_close = closes[-2]
            prev_last_fractal_high = last_fractal_highs[-2]
            prev_last_fractal_low = last_fractal_lows[-2]

            if prev_last_fractal_high is not None and prev_close > prev_last_fractal_high:
                return Signal(action="buy", strength=1.0, confidence=0.8)

            if self.exit_on_fractal_low and prev_last_fractal_low is not None and prev_close < prev_last_fractal_low:
                return Signal(action="sell", strength=1.0, confidence=0.8)

        return None