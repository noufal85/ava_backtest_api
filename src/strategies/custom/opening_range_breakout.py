"""opening_range_breakout — converted from trading-backtester."""
import polars as pl
import numpy as np
import math
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register

def atr(highs, lows, closes, period):
    """ATR from lists."""
    trs = []
    for i in range(len(highs)):
        tr = highs[i] - lows[i]
        if i > 0:
            tr = max(tr, abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    result = []
    for i in range(len(trs)):
        if i < period - 1:
            result.append(None)
        else:
            result.append(sum(trs[i-period+1:i+1]) / period)
    return result


@register
class OpeningRangeBreakout(BaseStrategy):
    name = "opening_range_breakout"
    name: str = "opening_range_breakout"
    version: str = "1.0.0"
    description: str = "Breakout: buy when open > prev high, exit at close or ATR stop"
    category: str = "momentum"
    tags: list[str] = ["breakout", "opening range"]

    def __init__(self, atr_period: int = 14, stop_atr_mult: float = 1.0):
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult

    def get_warmup_periods(self) -> int:
        return self.atr_period + 1

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()

        if historical_data.is_empty() or current_bar.is_empty():
            return None

        combined_data = pl.concat([historical_data, current_bar])
        opens = combined_data["open"].to_list()
        highs = combined_data["high"].to_list()
        lows = combined_data["low"].to_list()
        closes = combined_data["close"].to_list()

        if len(highs) < 2:
            return None

        atr_values = atr(highs, lows, closes, self.atr_period)
        atr_val = atr_values[-1]

        prev_high = highs[-2]
        prev_low = lows[-2]
        current_open = opens[-1]
        current_close = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]

        if current_open > prev_high:
            if atr_val is None:
                return None

            stop_price = current_open - (self.stop_atr_mult * atr_val)
            if current_low < stop_price:
                exit_price = stop_price
            else:
                exit_price = current_close

            return Signal(
                action="buy",
                strength=1.0,
                confidence=1.0,
                metadata={
                    "prev_high": prev_high,
                    "atr": atr_val,
                    "stop_price": stop_price,
                    "exit_price": exit_price
                }
            )
        elif current_open < prev_low:
            if atr_val is None:
                return None

            stop_price = current_open + (self.stop_atr_mult * atr_val)
            if current_high > stop_price:
                exit_price = stop_price
            else:
                exit_price = current_close

            return Signal(
                action="sell",
                strength=1.0,
                confidence=1.0,
                metadata={
                    "prev_low": prev_low,
                    "atr": atr_val,
                    "stop_price": stop_price,
                    "exit_price": exit_price
                }
            )
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "atr_period": {
                    "type": "integer",
                    "default": 14,
                    "description": "ATR calculation period"
                },
                "stop_atr_mult": {
                    "type": "number",
                    "default": 1.0,
                    "description": "ATR multiplier for stop loss"
                }
            },
            "required": []
        }