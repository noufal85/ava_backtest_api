"""tv_hull_ma_crossover — converted from trading-backtester."""
import polars as pl
import numpy as np
import math
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register

@register
class TvHullMaCrossover(BaseStrategy):
    name: str = "tv_hull_ma_crossover"
    version: str = "1.0.0"
    description: str = "Hull MA Crossover: Educational trend following with Hull Moving Averages"
    category: str = "trend"
    tags: list[str] = []

    def __init__(self, fast_length: int = 9, slow_length: int = 21):
        self.fast_length = fast_length
        self.slow_length = slow_length

    def get_parameter_schema(self) -> dict:
        return {
            "fast_length": {"type": "integer", "default": 9, "minimum": 1},
            "slow_length": {"type": "integer", "default": 21, "minimum": 2},
        }

    def get_warmup_periods(self) -> int:
        return max(self.fast_length, self.slow_length) + 5

    def _hull_ma(self, series: list[float], length: int) -> list[float]:
        """Calculate Hull Moving Average."""
        if length < 2:
            return series
        
        # HMA calculation: WMA(2*WMA(src, period/2) - WMA(src, period), sqrt(period))
        half_period = max(1, length // 2)
        sqrt_period = max(1, int(np.sqrt(length)))
        
        # WMA function
        def wma(data: list[float], period: int) -> list[float]:
            if len(data) < period:
                return [math.nan] * len(data)
            weights = np.arange(1, period + 1)
            result = []
            for i in range(period, len(data) + 1):
                window = data[i-period:i]
                result.append(np.dot(window, weights) / weights.sum())
            return [math.nan] * (period - 1) + result
        
        # First calculate the two WMAs
        wma_half = wma(series, half_period)
        wma_full = wma(series, length)
        
        # Then calculate 2*WMA(half) - WMA(full)
        diff = [2 * h - f if not (math.isnan(h) or math.isnan(f)) else math.nan for h, f in zip(wma_half, wma_full)]
        
        # Finally apply WMA with sqrt(length)
        hma = wma(diff, sqrt_period)
        
        return hma

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()
        
        if len(closes) < max(self.fast_length, self.slow_length):
            return None

        fast_hma_values = self._hull_ma(closes, self.fast_length)
        slow_hma_values = self._hull_ma(closes, self.slow_length)

        if len(fast_hma_values) < 3 or len(slow_hma_values) < 3:
            return None

        fast_hma = fast_hma_values[-1]
        slow_hma = slow_hma_values[-1]
        prev_fast_hma = fast_hma_values[-2]
        prev_slow_hma = slow_hma_values[-2]
        prev2_fast_hma = fast_hma_values[-3]
        prev2_slow_hma = slow_hma_values[-3]

        if math.isnan(fast_hma) or math.isnan(slow_hma) or math.isnan(prev_fast_hma) or math.isnan(prev_slow_hma) or math.isnan(prev2_fast_hma) or math.isnan(prev2_slow_hma):
            return None

        bullish_cross = (prev_fast_hma > prev_slow_hma) and (prev2_fast_hma <= prev2_slow_hma)
        bearish_cross = (prev_fast_hma < prev_slow_hma) and (prev2_fast_hma >= prev2_slow_hma)

        if bullish_cross:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"fast_hma": fast_hma, "slow_hma": slow_hma})
        elif bearish_cross:
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"fast_hma": fast_hma, "slow_hma": slow_hma})

        return None