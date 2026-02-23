"""tv_tema_ribbon — converted from trading-backtester."""
import polars as pl
import numpy as np
import math
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register

@register
class TvTemaRibbon(BaseStrategy):
    name: str = "tv_tema_ribbon"
    version: str = "1.0.0"
    description: str = "TEMA Ribbon: Multiple TEMA alignment trend following"
    category: str = "trend"
    tags: list[str] = []

    def __init__(
        self,
        tema1_length: int = 8,
        tema2_length: int = 13,
        tema3_length: int = 21,
        tema4_length: int = 34,
    ):
        self.tema1_length = tema1_length
        self.tema2_length = tema2_length
        self.tema3_length = tema3_length
        self.tema4_length = tema4_length

    def get_parameter_schema(self) -> dict:
        return {
            "tema1_length": {"type": "integer", "default": 8},
            "tema2_length": {"type": "integer", "default": 13},
            "tema3_length": {"type": "integer", "default": 21},
            "tema4_length": {"type": "integer", "default": 34},
        }

    def get_warmup_periods(self) -> int:
        return max(self.tema1_length, self.tema2_length, self.tema3_length, self.tema4_length) * 3

    def _tema(self, series: list[float], length: int) -> list[float]:
        """Calculate Triple Exponential Moving Average (TEMA)."""
        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        ema1 = self._ema(series, length)
        ema2 = self._ema(ema1, length)
        ema3 = self._ema(ema2, length)
        
        return [3 * e1 - 3 * e2 + e3 for e1, e2, e3 in zip(ema1, ema2, ema3)]

    def _ema(self, series: list[float], length: int) -> list[float]:
        """Calculate Exponential Moving Average (EMA)."""
        alpha = 2 / (length + 1)
        ema = [series[0]]  # Initialize EMA with the first value
        for i in range(1, len(series)):
            ema.append(alpha * series[i] + (1 - alpha) * ema[-1])
        return ema

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        close_prices = df["close"].to_list()

        if len(close_prices) < max(self.tema1_length, self.tema2_length, self.tema3_length, self.tema4_length) * 3:
            return None

        tema1 = self._tema(close_prices, self.tema1_length)
        tema2 = self._tema(close_prices, self.tema2_length)
        tema3 = self._tema(close_prices, self.tema3_length)
        tema4 = self._tema(close_prices, self.tema4_length)

        if len(tema1) < 3 or len(tema2) < 3 or len(tema3) < 3 or len(tema4) < 3:
            return None

        prev_tema1 = tema1[-2]
        prev_tema2 = tema2[-2]
        prev_tema3 = tema3[-2]
        prev_tema4 = tema4[-2]

        prev2_tema1 = tema1[-3]
        prev2_tema2 = tema2[-3]
        prev2_tema3 = tema3[-3]
        prev2_tema4 = tema4[-3]

        current_all_bull = (
            (prev_tema1 > prev_tema2) and 
            (prev_tema2 > prev_tema3) and 
            (prev_tema3 > prev_tema4)
        )
        
        current_all_bear = (
            (prev_tema1 < prev_tema2) and 
            (prev_tema2 < prev_tema3) and 
            (prev_tema3 < prev_tema4)
        )
        
        previous_all_bull = (
            (prev2_tema1 > prev2_tema2) and 
            (prev2_tema2 > prev2_tema3) and 
            (prev2_tema3 > prev2_tema4)
        )
        
        previous_all_bear = (
            (prev2_tema1 < prev2_tema2) and 
            (prev2_tema2 < prev2_tema3) and 
            (prev2_tema3 < prev2_tema4)
        )

        long_entry = current_all_bull and (not previous_all_bull)
        #short_entry = current_all_bear and (not previous_all_bear)

        if long_entry:
            return Signal(
                action="buy",
                strength=1.0,
                confidence=0.8,
                metadata={
                    "tema1": tema1[-1],
                    "tema2": tema2[-1],
                    "tema3": tema3[-1],
                    "tema4": tema4[-1],
                },
            )
        #elif short_entry:
        #    return Signal(
        #        action="sell",
        #        strength=1.0,
        #        confidence=0.8,
        #        metadata={
        #            "tema1": tema1[-1],
        #            "tema2": tema2[-1],
        #            "tema3": tema3[-1],
        #            "tema4": tema4[-1],
        #        },
        #    )
        else:
            return None