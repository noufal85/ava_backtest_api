"""tv_mfi — converted from trading-backtester."""
import polars as pl
import numpy as np
import math
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register

@register
class TvMfiStrategy(BaseStrategy):
    name = "tv_mfi"
    name: str = "tv_mfi"
    version: str = "1.0.0"
    description: str = "MFI Strategy: Uses Money Flow Index for volume-weighted momentum signals"
    category: str = "multi_factor"
    tags: list[str] = []

    def __init__(self, mfi_length: int = 14, overbought: float = 80.0, oversold: float = 20.0):
        self.mfi_length = mfi_length
        self.overbought = overbought
        self.oversold = oversold

    def get_warmup_periods(self) -> int:
        return self.mfi_length + 2  # Add a small buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.mfi_length + 2:
            return None

        high = df["high"].to_list()
        low = df["low"].to_list()
        close = df["close"].to_list()
        volume = df["volume"].to_list()

        typical_price = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]
        money_flow = [tp * v for tp, v in zip(typical_price, volume)]

        price_change = [typical_price[i] - typical_price[i - 1] if i > 0 else 0 for i in range(len(typical_price))]
        positive_mf = [mf if pc > 0 else 0 for pc, mf in zip(price_change, money_flow)]
        negative_mf = [mf if pc < 0 else 0 for pc, mf in zip(price_change, money_flow)]

        positive_mf_sum = []
        negative_mf_sum = []

        for i in range(len(positive_mf)):
            if i < self.mfi_length:
                positive_mf_sum.append(np.nan)
                negative_mf_sum.append(np.nan)
            else:
                positive_mf_sum.append(sum(positive_mf[i - self.mfi_length:i]))
                negative_mf_sum.append(sum(negative_mf[i - self.mfi_length:i]))

        mf_ratio = [p / n if n != 0 else 0 for p, n in zip(positive_mf_sum, negative_mf_sum)]
        mfi = [100 - (100 / (1 + ratio)) if ratio != 0 else 50 for ratio in mf_ratio]

        # Use shifted MFI to avoid look-ahead bias
        if len(mfi) < 3:
            return None

        prev_mfi = mfi[-2]
        prev2_mfi = mfi[-3]

        # MFI crossovers
        mfi_cross_above_os = (prev2_mfi <= self.oversold) and (prev_mfi > self.oversold)
        mfi_cross_below_ob = (prev2_mfi >= self.overbought) and (prev_mfi < self.overbought)

        if mfi_cross_above_os:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"mfi": mfi[-1]})
        #if mfi_cross_below_ob:
        #    return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"mfi": mfi[-1]})

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "mfi_length": {
                    "type": "integer",
                    "default": 14,
                    "description": "MFI calculation period"
                },
                "overbought": {
                    "type": "number",
                    "default": 80.0,
                    "description": "Overbought threshold"
                },
                "oversold": {
                    "type": "number",
                    "default": 20.0,
                    "description": "Oversold threshold"
                }
            },
            "required": ["mfi_length", "overbought", "oversold"]
        }