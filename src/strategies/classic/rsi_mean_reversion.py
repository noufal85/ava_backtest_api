"""Strategy 2: RSI Mean Reversion — buy oversold, sell overbought."""
import polars as pl

from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register


def _compute_rsi_wilder(closes: list[float], period: int) -> list[float]:
    """Compute RSI using Wilder smoothing (exponential moving average of gains/losses)."""
    if len(closes) < period + 1:
        return []

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Seed: simple average of first `period` deltas
    gains = [max(d, 0.0) for d in deltas[:period]]
    losses = [max(-d, 0.0) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    rsi_values: list[float] = []

    # First RSI value
    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100.0 - 100.0 / (1.0 + rs))

    # Wilder smoothing for subsequent values
    for i in range(period, len(deltas)):
        d = deltas[i]
        gain = max(d, 0.0)
        loss = max(-d, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - 100.0 / (1.0 + rs))

    return rsi_values


@register
class RSIMeanReversion(BaseStrategy):
    name = "rsi_mean_reversion"
    version = "2.0.0"
    description = "Buy when RSI crosses below oversold, exit when overbought or max hold."
    category = "mean_reversion"
    tags = ["mean_reversion", "rsi", "beginner"]

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        max_hold_days: int = 20,
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.max_hold_days = max_hold_days
        self._bars_held: int = 0
        self._in_position: bool = False

    def get_warmup_periods(self) -> int:
        # Need rsi_period + 1 prices to get first RSI, plus 1 more for crossing detection
        return self.rsi_period + 2

    def generate_signal(self, window) -> Signal | None:
        hist = window.historical()
        cur = window.current_bar()
        combined = pl.concat([hist, cur])
        closes = combined["close"].to_list()

        rsi_values = _compute_rsi_wilder(closes, self.rsi_period)
        if len(rsi_values) < 2:
            return None

        rsi_now = rsi_values[-1]
        rsi_prev = rsi_values[-2]

        if self._in_position:
            self._bars_held += 1
            # Exit: RSI > overbought OR held for max_hold_days
            if rsi_now > self.overbought or self._bars_held >= self.max_hold_days:
                self._in_position = False
                self._bars_held = 0
                return Signal(
                    action="sell",
                    strength=1.0,
                    confidence=0.7,
                    metadata={"rsi": rsi_now, "exit_reason": "overbought" if rsi_now > self.overbought else "max_hold"},
                )
        else:
            # Entry: RSI crosses below oversold (prev >= oversold, now < oversold)
            if rsi_prev >= self.oversold and rsi_now < self.oversold:
                self._in_position = True
                self._bars_held = 0
                return Signal(
                    action="buy",
                    strength=1.0,
                    confidence=0.8,
                    metadata={"rsi": rsi_now},
                )

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "rsi_period": {"type": "integer", "default": 14, "min": 5, "max": 50, "step": 1},
            "oversold": {"type": "number", "default": 30, "min": 10, "max": 40, "step": 5},
            "overbought": {"type": "number", "default": 70, "min": 60, "max": 90, "step": 5},
            "max_hold_days": {"type": "integer", "default": 20, "min": 5, "max": 60, "step": 1},
        }
