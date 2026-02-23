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
    """Calculate Average True Range (ATR)."""
    tr_values = []
    for i in range(len(high)):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr_values.append(tr)
    
    atr_values = []
    for i in range(len(tr_values)):
        if i < period:
            atr_values.append(np.nan)
        else:
            atr_values.append(sum(tr_values[i-period+1:i+1]) / period)
    return atr_values

def supertrend(high: list[float], low: list[float], close: list[float], period: int = 10, multiplier: float = 3.0) -> tuple[list[float], list[int]]:
    """Calculate Supertrend indicator."""
    hl2 = [(h + l) / 2 for h, l in zip(high, low)]
    atr_values = atr(high, low, close, period=period)
    
    # Calculate basic upper and lower bands
    basic_upper_band = [h + (multiplier * a) if not math.isnan(a) else np.nan for h, a in zip(hl2, atr_values)]
    basic_lower_band = [h - (multiplier * a) if not math.isnan(a) else np.nan for h, a in zip(hl2, atr_values)]
    
    # Initialize final bands
    final_upper_band = [np.nan] * len(high)
    final_lower_band = [np.nan] * len(high)
    supertrend_values = [np.nan] * len(high)
    direction = [0] * len(high)
    
    for i in range(len(high)):
        if i == 0:
            final_upper_band[i] = basic_upper_band[i]
            final_lower_band[i] = basic_lower_band[i]
            # Initialize direction based on close vs bands
            if close[i] <= final_lower_band[i]:
                direction[i] = 1  # Bearish
                supertrend_values[i] = final_upper_band[i]
            else:
                direction[i] = -1  # Bullish
                supertrend_values[i] = final_lower_band[i]
        else:
            # Calculate final upper band
            if (basic_upper_band[i] < final_upper_band[i-1] or 
                close[i-1] > final_upper_band[i-1]):
                final_upper_band[i] = basic_upper_band[i]
            else:
                final_upper_band[i] = final_upper_band[i-1]
            
            # Calculate final lower band
            if (basic_lower_band[i] > final_lower_band[i-1] or 
                close[i-1] < final_lower_band[i-1]):
                final_lower_band[i] = basic_lower_band[i]
            else:
                final_lower_band[i] = final_lower_band[i-1]
            
            # Determine direction and supertrend value
            if (direction[i-1] == 1 and close[i] > final_upper_band[i]) or \
               (direction[i-1] == -1 and close[i] < final_lower_band[i]):
                # Direction change
                if direction[i-1] == 1:
                    direction[i] = -1  # Change to bullish
                    supertrend_values[i] = final_lower_band[i]
                else:
                    direction[i] = 1   # Change to bearish
                    supertrend_values[i] = final_upper_band[i]
            else:
                # No direction change
                direction[i] = direction[i-1]
                if direction[i] == 1:  # Bearish
                    supertrend_values[i] = final_upper_band[i]
                else:  # Bullish
                    supertrend_values[i] = final_lower_band[i]
    
    return supertrend_values, direction

@register
class TvDoubleSupertrend(BaseStrategy):
    name = "tv_double_supertrend"
    version = "1.0.0"
    description = "Double Supertrend: Fast entry, slow exit with Supertrend indicators"
    category = "trend"
    tags = ["trend", "supertrend", "double supertrend"]

    def __init__(
        self,
        st1_atr_period: int = 10,
        st1_factor: float = 3.0,
        st2_atr_period: int = 10,
        st2_factor: float = 5.0,
        tp_type: str = "Supertrend",
        tp_percent: float = 0.015,
        sl_percent: float = 0.10,
    ):
        self.st1_atr_period = st1_atr_period
        self.st1_factor = st1_factor
        self.st2_atr_period = st2_atr_period
        self.st2_factor = st2_factor
        self.tp_type = tp_type
        self.tp_percent = tp_percent
        self.sl_percent = sl_percent

    def get_parameter_schema(self) -> dict:
        return {
            "st1_atr_period": {"type": "integer", "default": 10},
            "st1_factor": {"type": "number", "default": 3.0},
            "st2_atr_period": {"type": "integer", "default": 10},
            "st2_factor": {"type": "number", "default": 5.0},
            "tp_type": {"type": "string", "default": "Supertrend", "enum": ["Supertrend", "Percent"]},
            "tp_percent": {"type": "number", "default": 0.015},
            "sl_percent": {"type": "number", "default": 0.10},
        }

    def get_warmup_periods(self) -> int:
        return max(self.st1_atr_period, self.st2_atr_period) + 20

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        high = df["high"].to_list()
        low = df["low"].to_list()
        close = df["close"].to_list()

        st1_value, st1_direction = supertrend(high, low, close, period=self.st1_atr_period, multiplier=self.st1_factor)
        st2_value, st2_direction = supertrend(high, low, close, period=self.st2_atr_period, multiplier=self.st2_factor)

        if len(st1_direction) < 2 or len(st2_direction) < 2:
            return None

        prev_st1_direction = st1_direction[-2]
        prev_st2_direction = st2_direction[-2]
        current_st1_direction = st1_direction[-1]
        current_st2_direction = st2_direction[-1]

        long_entry = prev_st1_direction == -1 and current_st1_direction == 1
        long_exit = prev_st2_direction == -1 and current_st2_direction == 1

        if long_entry:
            return Signal(
                action="buy",
                strength=1.0,
                confidence=0.8,
                metadata={
                    "st1_value": st1_value[-1],
                    "st2_value": st2_value[-1],
                },
            )
        elif long_exit:
            return Signal(
                action="sell",
                strength=1.0,
                confidence=0.8,
                metadata={
                    "st1_value": st1_value[-1],
                    "st2_value": st2_value[-1],
                },
            )
        else:
            return None