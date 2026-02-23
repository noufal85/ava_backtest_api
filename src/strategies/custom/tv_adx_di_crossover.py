from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

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

def calculate_true_range(high: list[float], low: list[float], close: list[float]) -> list[float]:
    """Calculates the True Range (TR) for each period."""
    tr = [0.0]  # Initialize with a default value for the first period
    for i in range(1, len(high)):
        high_i = high[i]
        low_i = low[i]
        close_prev = close[i - 1]

        tr_val = max(
            high_i - low_i,
            abs(high_i - close_prev),
            abs(low_i - close_prev),
        )
        tr.append(tr_val)
    return tr

def calculate_directional_movement(high: list[float], low: list[float]) -> tuple[list[float], list[float]]:
    """Calculates +DM and -DM."""
    plus_dm = [0.0]
    minus_dm = [0.0]

    for i in range(1, len(high)):
        move_up = high[i] - high[i - 1]
        move_down = low[i - 1] - low[i]

        if move_up > move_down and move_up > 0:
            plus_dm.append(move_up)
        else:
            plus_dm.append(0.0)

        if move_down > move_up and move_down > 0:
            minus_dm.append(move_down)
        else:
            minus_dm.append(0.0)

    return plus_dm, minus_dm

def calculate_smoothed_values(values: list[float], period: int) -> list[float]:
    """Calculates smoothed values using Welles Wilder's smoothing method."""
    smoothed = [0.0] * len(values)
    smoothed[period] = sum(values[1:period+1])  # Adjusted indexing

    for i in range(period + 1, len(values)):  # Adjusted indexing
        smoothed[i] = smoothed[i - 1] - (smoothed[i - 1] / period) + values[i]
    return smoothed

def directional_indicators(df: pl.DataFrame, period: int) -> dict[str, list[float]]:
    """Calculates +DI and -DI."""
    high = df["high"].to_list()
    low = df["low"].to_list()
    close = df["close"].to_list()

    true_range = calculate_true_range(high, low, close)
    plus_dm, minus_dm = calculate_directional_movement(high, low)

    smoothed_tr = calculate_smoothed_values(true_range, period)
    smoothed_plus_dm = calculate_smoothed_values(plus_dm, period)
    smoothed_minus_dm = calculate_smoothed_values(minus_dm, period)

    di_plus = [
        (100 * smoothed_plus_dm[i] / smoothed_tr[i]) if smoothed_tr[i] != 0 else 0
        for i in range(len(high))
    ]
    di_minus = [
        (100 * smoothed_minus_dm[i] / smoothed_tr[i]) if smoothed_tr[i] != 0 else 0
        for i in range(len(high))
    ]

    return {"di_plus": di_plus, "di_minus": di_minus}

def calculate_adx(df: pl.DataFrame, period: int) -> list[float]:
    """Calculates the Average Directional Index (ADX)."""
    di_result = directional_indicators(df, period)
    di_plus = di_result["di_plus"]
    di_minus = di_result["di_minus"]

    dx = [
        (abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i]) * 100)
        if (di_plus[i] + di_minus[i]) != 0
        else 0
        for i in range(len(di_plus))
    ]

    adx = [0.0] * len(dx)
    adx[2 * period - 1] = sum(dx[period:2 * period]) / period

    for i in range(2 * period, len(dx)):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx

@register
class TvAdxDiCrossover(BaseStrategy):
    name = "tv_adx_di_crossover"
    version = "1.0.0"
    description = "ADX DI Crossover: Directional movement crossovers with ADX confirmation"
    category = "multi_factor"
    tags = ["trend", "adx", "di"]

    def __init__(self, adx_len: int = 14, adx_thresh: int = 25):
        self.adx_len = adx_len
        self.adx_thresh = adx_thresh

    def get_parameter_schema(self) -> dict:
        return {
            "adx_len": {"type": "integer", "default": 14, "minimum": 1},
            "adx_thresh": {"type": "integer", "default": 25, "minimum": 0, "maximum": 100},
        }

    def get_warmup_periods(self) -> int:
        return self.adx_len * 3 # Increased buffer

    def generate_signal(self, window) -> Optional[Signal]:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.get_warmup_periods():
            return None

        # Calculate ADX and Directional Movement Indicators
        adx_values = calculate_adx(df, self.adx_len)
        di_result = directional_indicators(df, self.adx_len)
        di_plus = di_result["di_plus"]
        di_minus = di_result["di_minus"]

        # Get the last values
        prev_di_plus = di_plus[-2]
        prev_di_minus = di_minus[-2]
        di_plus_val = di_plus[-1]
        di_minus_val = di_minus[-1]
        adx_val = adx_values[-2]

        # Crossover conditions
        di_plus_crossover = (prev_di_plus <= prev_di_minus) and (di_plus_val > di_minus_val)
        di_plus_crossunder = (prev_di_plus >= prev_di_minus) and (di_plus_val < di_minus_val)

        # Entry conditions with ADX filter
        if di_plus_crossover and (adx_val > self.adx_thresh):
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"adx": adx_val, "di_plus": di_plus_val, "di_minus": di_minus_val})
        elif di_plus_crossunder and (adx_val > self.adx_thresh):
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"adx": adx_val, "di_plus": di_plus_val, "di_minus": di_minus_val})

        return None