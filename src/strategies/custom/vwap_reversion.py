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

def calculate_vwap(df: pl.DataFrame) -> float:
    """Calculates the Volume Weighted Average Price (VWAP)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    weighted_price = typical_price * df["volume"]
    total_value = weighted_price.sum()
    total_volume = df["volume"].sum()
    if total_volume == 0:
        return np.nan
    return total_value / total_volume

@register
class VwapReversion(BaseStrategy):
    name = "vwap_reversion"
    version = "1.1.0"
    description = "Mean reversion: buy/short when price deviates from VWAP, exit on return"
    category = "mean_reversion"
    tags = ["vwap", "mean_reversion"]

    def __init__(self, vwap_period: int = 20, entry_deviation: float = 0.02, exit_target: float = 0.005, max_hold: int = 5):
        self.vwap_period = vwap_period
        self.entry_deviation = entry_deviation
        self.exit_target = exit_target
        self.max_hold = max_hold
        self.hold_count = 0
        self.position_direction = None
        self.entry_price = None
        self.entry_vwap = None

    def get_warmup_periods(self) -> int:
        return self.vwap_period + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.vwap_period:
            return None

        vwap = calculate_vwap(df)

        if np.isnan(vwap):
            return None

        current_close = current_bar["close"][0]

        if self.position_direction is None:
            if current_close < vwap * (1 - self.entry_deviation):
                self.position_direction = "long"
                self.entry_price = current_close
                self.entry_vwap = vwap
                self.hold_count = 1
                return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"vwap": vwap, "close": current_close})
            elif current_close > vwap * (1 + self.entry_deviation):
                self.position_direction = "short"
                self.entry_price = current_close
                self.entry_vwap = vwap
                self.hold_count = 1
                return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"vwap": vwap, "close": current_close})
            else:
                return None
        else:
            self.hold_count += 1
            if self.position_direction == "long":
                if abs(current_close - vwap) / vwap <= self.exit_target or self.hold_count > self.max_hold:
                    self.position_direction = None
                    self.entry_price = None
                    self.entry_vwap = None
                    self.hold_count = 0
                    return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"reason": "vwap_reversion" if abs(current_close - vwap) / vwap <= self.exit_target else "max_hold", "vwap": vwap, "close": current_close})
                elif current_close > vwap * (1 + self.entry_deviation):
                    self.position_direction = "short"
                    self.entry_price = current_close
                    self.entry_vwap = vwap
                    self.hold_count = 1
                    return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"reason": "signal_reversal", "vwap": vwap, "close": current_close})
                else:
                    return None
            elif self.position_direction == "short":
                if abs(current_close - vwap) / vwap <= self.exit_target or self.hold_count > self.max_hold:
                    self.position_direction = None
                    self.entry_price = None
                    self.entry_vwap = None
                    self.hold_count = 0
                    return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"reason": "vwap_reversion" if abs(current_close - vwap) / vwap <= self.exit_target else "max_hold", "vwap": vwap, "close": current_close})
                elif current_close < vwap * (1 - self.entry_deviation):
                    self.position_direction = "long"
                    self.entry_price = current_close
                    self.entry_vwap = vwap
                    self.hold_count = 1
                    return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"reason": "signal_reversal", "vwap": vwap, "close": current_close})
                else:
                    return None
            else:
                return None

    def get_parameter_schema(self) -> dict:
        return {
            "vwap_period": {"type": "integer", "default": 20, "minimum": 1},
            "entry_deviation": {"type": "number", "default": 0.02, "minimum": 0.0},
            "exit_target": {"type": "number", "default": 0.005, "minimum": 0.0},
            "max_hold": {"type": "integer", "default": 5, "minimum": 1},
        }