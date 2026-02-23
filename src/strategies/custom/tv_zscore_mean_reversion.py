"""tv_zscore_mean_reversion strategy — converted from trading-backtester."""
import polars as pl
import numpy as np
import math
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register


def sma(values, period):
    """SMA from list."""
    result = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(None)
        else:
            result.append(sum(values[i-period+1:i+1])/period)
    return result

@register
class TvZscoreMeanReversion(BaseStrategy):
    name: str = "tv_zscore_mean_reversion"
    version: str = "1.0.0"
    description: str = "Statistical mean reversion using Z-score thresholds"
    category: str = "mean_reversion"
    tags: list[str] = []

    def __init__(self, length: int = 20, z_entry: float = 2.0, z_exit: float = 0.5):
        self.length = length
        self.z_entry = z_entry
        self.z_exit = z_exit

    def get_warmup_periods(self) -> int:
        return self.length + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.length:
            return None

        closes = df["close"]
        
        # Calculate SMA
        sma_values = sma(closes, self.length)
        
        # Calculate Standard Deviation
        std_values = []
        for i in range(len(closes)):
            if i < self.length - 1:
                std_values.append(None)
            else:
                window_std = closes[i - self.length + 1:i + 1].std()
                std_values.append(window_std)
        std = pl.Series(std_values)
        
        # Calculate Z-score
        zscore_values = []
        for i in range(len(closes)):
            if std[i] is not None and std[i] > 0:
                zscore_values.append((closes[i] - sma_values[i]) / std[i])
            else:
                zscore_values.append(0.0)
        zscore = pl.Series(zscore_values)

        current_zscore = zscore[-1]
        previous_zscore = zscore[-2] if len(zscore) > 1 else None

        if previous_zscore is None:
            return None

        # Entry signals
        if previous_zscore > -self.z_entry and current_zscore <= -self.z_entry:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"zscore": current_zscore})
        elif previous_zscore < self.z_entry and current_zscore >= self.z_entry:
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"zscore": current_zscore})
        elif abs(current_zscore) < self.z_exit:
            # Check if there's an existing long or short position to close
            # This requires knowledge of the current position, which is not available in this context.
            # Returning None here means no action if Z-score returns to neutral.
            return None
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "length": {
                    "type": "integer",
                    "default": 20,
                    "description": "Period for SMA and standard deviation"
                },
                "z_entry": {
                    "type": "number",
                    "default": 2.0,
                    "description": "Z-score entry threshold"
                },
                "z_exit": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Z-score exit threshold"
                }
            },
            "required": ["length", "z_entry", "z_exit"]
        }