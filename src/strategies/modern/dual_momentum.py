"""Strategy 7: Dual Momentum (Antonacci) — absolute momentum with rebalancing."""
import polars as pl

from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register


@register
class DualMomentum(BaseStrategy):
    name = "dual_momentum"
    version = "2.0.0"
    description = "Buy if lookback return > 0 (cash), rebalance monthly."
    category = "momentum"
    tags = ["momentum", "dual_momentum", "rebalance", "advanced"]

    def __init__(self, lookback_months: int = 12, rebalance_frequency: int = 21):
        self.lookback_months = lookback_months
        self.rebalance_frequency = rebalance_frequency
        # Approximate trading days per month
        self._lookback_bars = lookback_months * 21
        self._bar_count: int = 0
        self._in_position: bool = False

    def get_warmup_periods(self) -> int:
        return self._lookback_bars + 1

    def generate_signal(self, window) -> Signal | None:
        hist = window.historical()
        cur = window.current_bar()
        combined = pl.concat([hist, cur])
        closes = combined["close"].to_list()

        if len(closes) < self._lookback_bars + 1:
            return None

        self._bar_count += 1

        # Only rebalance every rebalance_frequency bars
        if self._bar_count % self.rebalance_frequency != 1 and self._bar_count != 1:
            return None

        current_price = closes[-1]
        lookback_price = closes[-self._lookback_bars - 1]

        # Target return vs cash (0%)
        target_return = (current_price - lookback_price) / lookback_price

        if target_return > 0 and not self._in_position:
            self._in_position = True
            return Signal(
                action="buy",
                strength=min(abs(target_return) * 5, 1.0),
                confidence=0.7,
                metadata={"target_return": target_return, "lookback_bars": self._lookback_bars},
            )
        elif target_return <= 0 and self._in_position:
            self._in_position = False
            return Signal(
                action="sell",
                strength=1.0,
                confidence=0.7,
                metadata={"target_return": target_return, "lookback_bars": self._lookback_bars},
            )

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "lookback_months": {"type": "integer", "default": 12, "min": 1, "max": 24, "step": 1},
            "rebalance_frequency": {"type": "integer", "default": 21, "min": 5, "max": 63, "step": 1},
        }
