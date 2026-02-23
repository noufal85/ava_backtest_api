"""earnings_gap_fade strategy — converted from trading-backtester."""
import polars as pl
import numpy as np
import math
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register

@register
class EarningsGapFade(BaseStrategy):
    name: str = "earnings_gap_fade"
    version: str = "1.0.0"
    description: str = "Fade large overnight gaps expecting mean reversion to previous close"
    category: str = "multi_factor"
    tags: list[str] = []

    def __init__(self, min_gap_pct: float = 0.03, direction: str = "fade_down", stop_pct: float = 0.02):
        self.min_gap_pct = min_gap_pct
        self.direction = direction
        self.stop_pct = stop_pct

    def get_parameter_schema(self) -> dict:
        return {
            "min_gap_pct": {"type": "number", "default": 0.03, "description": "Minimum gap size to trigger"},
            "direction": {"type": "string", "default": "fade_down", "enum": ["fade_up", "fade_down", "both"], "description": "Trade direction"},
            "stop_pct": {"type": "number", "default": 0.02, "description": "Stop-loss as pct of entry price"}
        }

    def get_warmup_periods(self) -> int:
        return 2  # 1 for prev_close, plus a small buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()

        if historical_data.is_empty() or current_bar.is_empty():
            return None

        combined_data = pl.concat([historical_data, current_bar])

        if combined_data.height < 2:
            return None

        current_open = current_bar["open"][0]
        previous_close = historical_data["close"][-1]

        if previous_close == 0:
            return None

        gap_pct = (current_open - previous_close) / previous_close

        if math.isnan(gap_pct):
            return None

        if self.direction in ("fade_down", "both") and gap_pct <= -self.min_gap_pct:
            return Signal(action="buy", strength=abs(gap_pct), confidence=1.0, metadata={"gap_pct": gap_pct})

        if self.direction in ("fade_up", "both") and gap_pct >= self.min_gap_pct:
            return Signal(action="sell", strength=abs(gap_pct), confidence=1.0, metadata={"gap_pct": gap_pct})

        return None