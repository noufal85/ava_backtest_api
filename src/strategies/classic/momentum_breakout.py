"""Strategy 5: Momentum Breakout — channel breakout entry, MA exit."""
import polars as pl

from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register


@register
class MomentumBreakout(BaseStrategy):
    name = "momentum_breakout"
    version = "2.0.0"
    description = "Buy when close breaks above N-day high channel, exit below MA."
    category = "momentum"
    tags = ["momentum", "breakout", "intermediate"]

    def __init__(self, channel_period: int = 20, exit_ma_period: int = 20):
        self.channel_period = channel_period
        self.exit_ma_period = exit_ma_period
        self._in_position: bool = False

    def get_warmup_periods(self) -> int:
        return max(self.channel_period, self.exit_ma_period) + 1

    def generate_signal(self, window) -> Signal | None:
        hist = window.historical()
        cur = window.current_bar()
        combined = pl.concat([hist, cur])

        closes = combined["close"].to_list()
        highs = combined["high"].to_list()

        if len(closes) < max(self.channel_period, self.exit_ma_period) + 1:
            return None

        current_close = closes[-1]

        if self._in_position:
            # Exit: close < SMA(close, exit_ma_period)
            exit_ma = sum(closes[-self.exit_ma_period :]) / self.exit_ma_period
            if current_close < exit_ma:
                self._in_position = False
                return Signal(
                    action="sell",
                    strength=1.0,
                    confidence=0.7,
                    metadata={"close": current_close, "exit_ma": exit_ma},
                )
        else:
            # Entry: close > max(high[-channel_period:]) — excluding current bar
            channel_high = max(highs[-self.channel_period - 1 : -1])
            if current_close > channel_high:
                self._in_position = True
                return Signal(
                    action="buy",
                    strength=1.0,
                    confidence=0.8,
                    metadata={"close": current_close, "channel_high": channel_high},
                )

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "channel_period": {"type": "integer", "default": 20, "min": 5, "max": 60, "step": 1},
            "exit_ma_period": {"type": "integer", "default": 20, "min": 5, "max": 60, "step": 1},
        }
