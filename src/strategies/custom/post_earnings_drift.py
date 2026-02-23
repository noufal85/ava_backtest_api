from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import polars as pl
import numpy as np
import math

from src.core.strategy.registry import register

@dataclass
class Signal:
    action: str          # "buy" | "sell" | "hold"
    strength: float      # 0.0 – 1.0
    confidence: float    # 0.0 – 1.0
    metadata: dict = field(default_factory=dict)

class BaseStrategy(ABC):
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    category: str = ""   # trend | mean_reversion | momentum | multi_factor | volatility
    tags: list[str] = []

    @abstractmethod
    def get_warmup_periods(self) -> int: ...

    @abstractmethod
    def generate_signal(self, window) -> Signal | None:
        # window.current_bar()   → polars DataFrame row (current bar)
        # window.historical()    → polars DataFrame of all historical bars
        # Columns: open, high, low, close, volume (float64), timestamp (date)
        # Combine: pl.concat([window.historical(), window.current_bar()])
        # NEVER look ahead
        ...

    def get_parameter_schema(self) -> dict:
        return {}

def volume_sma(volume: list[float], period: int) -> list[float]:
    """Simple moving average for volume."""
    if len(volume) < period:
        return [None] * len(volume)
    sma = []
    for i in range(len(volume)):
        if i < period - 1:
            sma.append(None)
        else:
            sma.append(sum(volume[i - period + 1:i + 1]) / period)
    return sma

@register
class PostEarningsDriftStrategy(BaseStrategy):
    name = "post_earnings_drift"
    version = "1.0.0"
    description = "Momentum: buy on large gap-up / short on large gap-down with high volume (earnings proxy)"
    category = "multi_factor"
    tags = ["momentum", "gap", "volume"]

    def __init__(self, min_gap_pct: float = 0.03, min_volume_mult: float = 2.0, hold_days: int = 60, exit_vol_thresh: float = 0.5, profit_target: float = 0.10):
        self.min_gap_pct = min_gap_pct
        self.min_volume_mult = min_volume_mult
        self.hold_days = hold_days
        self.exit_vol_thresh = exit_vol_thresh
        self.profit_target = profit_target

    def get_warmup_periods(self) -> int:
        return 20 + 2  # 20 for volume_sma, plus a small buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if df.height < 21:
            return None

        closes = df["close"].to_list()
        opens = df["open"].to_list()
        volumes = df["volume"].to_list()

        vol_sma_20 = volume_sma(volumes, 20)
        
        prev_close = closes[-2]
        current_open = opens[-1]
        current_volume = volumes[-1]
        current_vol_sma = vol_sma_20[-1]

        if prev_close is None or current_vol_sma is None:
            return None

        gap = (current_open - prev_close) / prev_close if prev_close != 0 else 0.0
        vol_ratio = current_volume / current_vol_sma if current_vol_sma != 0 else 0.0

        returns_3d = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 and closes[-4] != 0 else 0.0

        # Entry conditions
        buy_mask = (gap > self.min_gap_pct) and (current_volume > self.min_volume_mult * current_vol_sma)
        short_mask = (gap < -self.min_gap_pct) and (current_volume > self.min_volume_mult * current_vol_sma)

        # Exit conditions
        momentum_fade = vol_ratio < self.exit_vol_thresh  # Volume dried up
        profit_target_hit = (
            (returns_3d > self.profit_target) or  # 3-day returns hit target 
            (returns_3d < -self.profit_target)   # Or hit loss limit
        )
        reverse_gap = (
            (gap > self.min_gap_pct) or    # New gap up (exit shorts)
            (gap < -self.min_gap_pct)     # New gap down (exit longs)
        )
        
        # Exit signal: momentum fade OR profit target OR reverse gap (but not on same bar as entry)
        exit_mask = (momentum_fade or profit_target_hit or reverse_gap)

        if buy_mask:
            return Signal(action="buy", strength=min(gap, 1.0), confidence=1.0, metadata={"gap": gap})
        elif short_mask:
            return Signal(action="sell", strength=min(abs(gap), 1.0), confidence=1.0, metadata={"gap": gap})
        elif exit_mask:
            return Signal(action="sell", strength=1.0, confidence=1.0)
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "min_gap_pct": {
                    "type": "number",
                    "description": "Minimum overnight gap to trigger (e.g., 0.03 for 3%)",
                    "default": 0.03,
                },
                "min_volume_mult": {
                    "type": "number",
                    "description": "Minimum volume multiple vs 20d avg",
                    "default": 2.0,
                },
                "hold_days": {
                    "type": "integer",
                    "description": "Fixed holding period in days",
                    "default": 60,
                },
                "exit_vol_thresh": {
                    "type": "number",
                    "description": "Exit when volume drops below this fraction of the average",
                    "default": 0.5,
                },
                "profit_target": {
                    "type": "number",
                    "description": "Take profit at this return percentage (e.g., 0.10 for 10%)",
                    "default": 0.10,
                },
            },
            "required": ["min_gap_pct", "min_volume_mult", "hold_days", "exit_vol_thresh", "profit_target"],
        }