"""Strategy 1: SMA Crossover — trend following."""
import polars as pl

from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register


@register
class SMACrossover(BaseStrategy):
    name = "sma_crossover"
    version = "2.0.0"
    description = "Buy when fast SMA crosses above slow SMA, sell on cross-under."
    category = "trend"
    tags = ["trend_following", "sma", "beginner"]

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be < slow_period ({slow_period})"
            )
        self.fast_period = fast_period
        self.slow_period = slow_period

    def get_warmup_periods(self) -> int:
        return self.slow_period + 1

    def generate_signal(self, window) -> Signal | None:
        hist = window.historical()
        cur = window.current_bar()
        if len(hist) < self.slow_period:
            return None

        combined = pl.concat([hist, cur])
        closes = combined["close"].to_list()

        fast_now = sum(closes[-self.fast_period :]) / self.fast_period
        fast_prev = sum(closes[-self.fast_period - 1 : -1]) / self.fast_period
        slow_now = sum(closes[-self.slow_period :]) / self.slow_period
        slow_prev = sum(closes[-self.slow_period - 1 : -1]) / self.slow_period

        # Crossover: fast crosses ABOVE slow (BUY)
        if fast_prev <= slow_prev and fast_now > slow_now:
            return Signal(
                action="buy",
                strength=1.0,
                confidence=0.8,
                metadata={"fast_sma": fast_now, "slow_sma": slow_now},
            )
        # Cross-under: fast crosses BELOW slow (SELL)
        if fast_prev >= slow_prev and fast_now < slow_now:
            return Signal(
                action="sell",
                strength=1.0,
                confidence=0.8,
                metadata={"fast_sma": fast_now, "slow_sma": slow_now},
            )
        return None

    def get_parameter_schema(self) -> dict:
        return {
            "fast_period": {
                "type": "integer",
                "default": 20,
                "min": 5,
                "max": 100,
                "step": 1,
            },
            "slow_period": {
                "type": "integer",
                "default": 50,
                "min": 10,
                "max": 300,
                "step": 1,
            },
        }
