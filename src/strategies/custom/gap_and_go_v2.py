from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import polars as pl
import numpy as np
import math

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

from src.core.strategy.registry import register

def atr(high: list[float], low: list[float], close: list[float], period: int) -> list[float]:
    tr = [0.0]  # Initialize with a default value
    for i in range(1, len(high)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))

    atr_values = [0.0] * len(high)
    if len(high) >= period:
        atr_values[period - 1] = sum(tr[0:period]) / period
        for i in range(period, len(high)):
            atr_values[i] = (atr_values[i - 1] * (period - 1) + tr[i]) / period
    return atr_values

def volume_sma(volume: list[float], period: int) -> list[float]:
    sma_values = [0.0] * len(volume)
    if len(volume) >= period:
        sma_values[period - 1] = sum(volume[0:period]) / period
        for i in range(period, len(volume)):
            sma_values[i] = (sma_values[i - 1] * (period - 1) + volume[i]) / period
    return sma_values

@register
class GapAndGoV2(BaseStrategy):
    name = "gap_and_go_v2"
    version = "1.0.0"
    description = "Gap and Go V2: trade opening gaps with volume confirmation"
    category = "momentum"
    tags = ["gap", "momentum", "volume"]

    def __init__(
        self,
        min_gap_pct: float = 0.02,
        max_gap_pct: float = 0.08,
        vol_mult: float = 1.5,
        vol_sma_period: int = 20,
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,
        max_hold: int = 3,
    ):
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.vol_mult = vol_mult
        self.vol_sma_period = vol_sma_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.max_hold = max_hold

    def get_warmup_periods(self) -> int:
        return max(self.vol_sma_period, self.atr_period) + 2

    def generate_signal(self, window) -> Signal | None:
        historical_df = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_df, current_bar])

        if len(df) < self.get_warmup_periods():
            return None

        closes = df["close"].to_list()
        opens = df["open"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        volumes = df["volume"].to_list()

        atrs = atr(highs, lows, closes, self.atr_period)
        vol_smas = volume_sma(volumes, self.vol_sma_period)

        # Use previous bar's gap info to signal for next bar entry
        if len(closes) < 2:
            return None

        prev_close = closes[-2]
        current_open = opens[-1]
        current_volume = volumes[-1]
        current_vol_sma = vol_smas[-1]

        gap_pct = (current_open - prev_close) / prev_close if prev_close != 0 else 0

        if (
            self.min_gap_pct < gap_pct < self.max_gap_pct
            and current_volume > (self.vol_mult * current_vol_sma)
            and prev_close > current_open
        ):
            return Signal(action="buy", strength=1.0, confidence=1.0)

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "min_gap_pct": {
                    "type": "number",
                    "default": 0.02,
                    "description": "Minimum gap up percentage",
                },
                "max_gap_pct": {
                    "type": "number",
                    "default": 0.08,
                    "description": "Maximum gap to avoid overextension",
                },
                "vol_mult": {
                    "type": "number",
                    "default": 1.5,
                    "description": "Volume must exceed this × avg volume",
                },
                "vol_sma_period": {
                    "type": "integer",
                    "default": 20,
                    "description": "Volume SMA lookback",
                },
                "atr_period": {
                    "type": "integer",
                    "default": 14,
                    "description": "ATR period",
                },
                "atr_stop_mult": {
                    "type": "number",
                    "default": 1.5,
                    "description": "ATR multiplier for stop",
                },
                "max_hold": {
                    "type": "integer",
                    "default": 3,
                    "description": "Max holding days",
                },
            },
            "required": [
                "min_gap_pct",
                "max_gap_pct",
                "vol_mult",
                "vol_sma_period",
                "atr_period",
                "atr_stop_mult",
                "max_hold",
            ],
        }