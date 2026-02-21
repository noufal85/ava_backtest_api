"""Strategy 4: Bollinger Bands Mean Reversion — buy below lower band, exit at middle."""
import math

import polars as pl

from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register


@register
class BollingerBands(BaseStrategy):
    name = "bollinger_bands"
    version = "2.0.0"
    description = "Buy when close < lower band, exit when close > middle band."
    category = "mean_reversion"
    tags = ["mean_reversion", "bollinger", "intermediate"]

    def __init__(self, bb_period: int = 20, bb_std_dev: float = 2.0):
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self._in_position: bool = False

    def get_parameter_schema(self) -> dict:
        return {
            "bb_period":  {"type": "integer", "default": 20,  "minimum": 5,   "maximum": 100, "description": "Bollinger Band period"},
            "bb_std_dev": {"type": "number",  "default": 2.0, "minimum": 0.5, "maximum": 4.0, "description": "Standard deviations for bands"},
        }

    def get_warmup_periods(self) -> int:
        return self.bb_period + 1

    def _compute_bands(self, closes: list[float]) -> tuple[float, float, float]:
        """Returns (lower, middle, upper) bands from the last bb_period closes."""
        window = closes[-self.bb_period :]
        middle = sum(window) / self.bb_period
        variance = sum((c - middle) ** 2 for c in window) / self.bb_period
        std = math.sqrt(variance)
        lower = middle - self.bb_std_dev * std
        upper = middle + self.bb_std_dev * std
        return lower, middle, upper

    def generate_signal(self, window) -> Signal | None:
        hist = window.historical()
        cur = window.current_bar()
        combined = pl.concat([hist, cur])
        closes = combined["close"].to_list()

        if len(closes) < self.bb_period:
            return None

        lower, middle, upper = self._compute_bands(closes)
        current_close = closes[-1]

        if self._in_position:
            # Exit: close > middle band (conservative exit)
            if current_close > middle:
                self._in_position = False
                return Signal(
                    action="sell",
                    strength=1.0,
                    confidence=0.7,
                    metadata={"close": current_close, "middle": middle, "upper": upper},
                )
        else:
            # Entry: close < lower band
            if current_close < lower:
                self._in_position = True
                return Signal(
                    action="buy",
                    strength=1.0,
                    confidence=0.8,
                    metadata={"close": current_close, "lower": lower, "middle": middle},
                )

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "bb_period": {"type": "integer", "default": 20, "min": 10, "max": 50, "step": 1},
            "bb_std_dev": {"type": "number", "default": 2.0, "min": 1.0, "max": 3.0, "step": 0.5},
        }
