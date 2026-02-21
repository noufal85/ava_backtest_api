"""Strategy 8: Opening Range Breakout — bullish breakout with fixed hold."""
import polars as pl

from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register


@register
class OpeningRangeBreakout(BaseStrategy):
    name = "opening_range_breakout"
    version = "2.0.0"
    description = "Buy on bullish breakout day (close > open + range*pct), exit after hold_bars."
    category = "momentum"
    tags = ["breakout", "orb", "short_term"]

    def __init__(self, range_pct: float = 0.3, hold_bars: int = 1):
        self.range_pct = range_pct
        self.hold_bars = hold_bars
        self._in_position: bool = False
        self._bars_held: int = 0

    def get_warmup_periods(self) -> int:
        return 2  # Minimal warmup — just need current bar

    def generate_signal(self, window) -> Signal | None:
        cur = window.current_bar()
        cur_open = cur["open"][0]
        cur_high = cur["high"][0]
        cur_low = cur["low"][0]
        cur_close = cur["close"][0]

        if self._in_position:
            self._bars_held += 1
            # Exit after exactly hold_bars bars
            if self._bars_held >= self.hold_bars:
                self._in_position = False
                self._bars_held = 0
                return Signal(
                    action="sell",
                    strength=1.0,
                    confidence=0.9,
                    metadata={"exit_reason": "hold_expired", "bars_held": self.hold_bars},
                )
        else:
            # Entry: close > open + (high - low) * range_pct on a bullish breakout day
            bar_range = cur_high - cur_low
            threshold = cur_open + bar_range * self.range_pct
            if cur_close > threshold and bar_range > 0:
                self._in_position = True
                self._bars_held = 0
                return Signal(
                    action="buy",
                    strength=min(((cur_close - threshold) / bar_range) * 2, 1.0),
                    confidence=0.7,
                    metadata={
                        "close": cur_close,
                        "threshold": threshold,
                        "bar_range": bar_range,
                    },
                )

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "range_pct": {"type": "number", "default": 0.3, "min": 0.1, "max": 0.8, "step": 0.05},
            "hold_bars": {"type": "integer", "default": 1, "min": 1, "max": 10, "step": 1},
        }
