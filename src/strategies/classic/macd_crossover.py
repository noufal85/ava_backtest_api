"""Strategy 3: MACD Crossover — trend following with MACD histogram."""
import polars as pl

from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register


def _ema(values: list[float], period: int) -> list[float]:
    """Compute EMA series. Returns list same length as input (first value = first input)."""
    if not values:
        return []
    multiplier = 2.0 / (period + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(v * multiplier + result[-1] * (1 - multiplier))
    return result


@register
class MACDCrossover(BaseStrategy):
    name = "macd_crossover"
    version = "2.0.0"
    description = "Buy when MACD crosses above signal line, sell on cross-under."
    category = "trend"
    tags = ["trend_following", "macd", "intermediate"]

    def __init__(
        self,
        fast_ema: int = 12,
        slow_ema: int = 26,
        signal_period: int = 9,
    ):
        if fast_ema >= slow_ema:
            raise ValueError(
                f"fast_ema ({fast_ema}) must be < slow_ema ({slow_ema})"
            )
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.signal_period = signal_period

    def get_warmup_periods(self) -> int:
        # Need slow_ema bars for EMA to stabilize + signal_period + 1 for crossing
        return self.slow_ema + self.signal_period + 1

    def generate_signal(self, window) -> Signal | None:
        hist = window.historical()
        cur = window.current_bar()
        combined = pl.concat([hist, cur])
        closes = combined["close"].to_list()

        if len(closes) < self.slow_ema + self.signal_period:
            return None

        fast_ema = _ema(closes, self.fast_ema)
        slow_ema = _ema(closes, self.slow_ema)

        # MACD line = fast EMA - slow EMA
        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]

        # Signal line = EMA of MACD line
        signal_line = _ema(macd_line, self.signal_period)

        macd_now = macd_line[-1]
        macd_prev = macd_line[-2]
        sig_now = signal_line[-1]
        sig_prev = signal_line[-2]

        # MACD crosses above signal → BUY
        if macd_prev <= sig_prev and macd_now > sig_now:
            return Signal(
                action="buy",
                strength=1.0,
                confidence=0.75,
                metadata={"macd": macd_now, "signal": sig_now, "histogram": macd_now - sig_now},
            )
        # MACD crosses below signal → SELL
        if macd_prev >= sig_prev and macd_now < sig_now:
            return Signal(
                action="sell",
                strength=1.0,
                confidence=0.75,
                metadata={"macd": macd_now, "signal": sig_now, "histogram": macd_now - sig_now},
            )
        return None

    def get_parameter_schema(self) -> dict:
        return {
            "fast_ema": {"type": "integer", "default": 12, "min": 5, "max": 50, "step": 1},
            "slow_ema": {"type": "integer", "default": 26, "min": 10, "max": 100, "step": 1},
            "signal_period": {"type": "integer", "default": 9, "min": 3, "max": 20, "step": 1},
        }
