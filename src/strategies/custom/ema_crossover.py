import polars as pl
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register
import numpy as np

@register
class EmaCrossover(BaseStrategy):
    name = "ema_crossover"
    version = "1.0.0"
    description = "Buy when 12-period EMA crosses above 26-period EMA, sell when it crosses below."
    category = "trend"
    tags = ["trend_following", "ema", "crossover"]

    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def get_warmup_periods(self) -> int:
        return self.slow_period + 1

    def get_parameter_schema(self) -> dict:
        return {
            "fast_period": {"type": "integer", "default": 12, "minimum": 5, "maximum": 50, "description": "Fast EMA period"},
            "slow_period": {"type": "integer", "default": 26, "minimum": 10, "maximum": 100, "description": "Slow EMA period"},
        }

    def calculate_ema(self, closes: list[float], period: int) -> list[float]:
        ema = [sum(closes[:period]) / period]
        k = 2 / (period + 1)
        for i in range(period, len(closes)):
            ema.append(closes[i] * k + ema[-1] * (1 - k))
        return ema

    def generate_signal(self, window) -> Signal | None:
        hist = window.historical()
        cur = window.current_bar()
        if len(hist) < self.slow_period:
            return None
        combined = pl.concat([hist, cur])
        closes = combined["close"].to_list()

        fast_ema = self.calculate_ema(closes, self.fast_period)
        slow_ema = self.calculate_ema(closes, self.slow_period)

        fast_ema_now = fast_ema[-1]
        fast_ema_prev = fast_ema[-2]
        slow_ema_now = slow_ema[-1]
        slow_ema_prev = slow_ema[-2]

        if fast_ema_prev <= slow_ema_prev and fast_ema_now > slow_ema_now:
            return Signal("buy", 1.0, 0.8, {"fast_ema": fast_ema_now, "slow_ema": slow_ema_now})
        if fast_ema_prev >= slow_ema_prev and fast_ema_now < slow_ema_now:
            return Signal("sell", 1.0, 0.8, {"fast_ema": fast_ema_now, "slow_ema": slow_ema_now})
        return None