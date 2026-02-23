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

def aroon(high: list[float], low: list[float], length: int = 14) -> tuple[list[float], list[float]]:
    """Aroon Up and Down indicators."""
    aroon_up = [np.nan] * len(high)
    aroon_down = [np.nan] * len(high)
    for i in range(length, len(high)):
        window_high = high[i-length:i+1]
        window_low = low[i-length:i+1]
        bars_since_high = length - window_high.index(max(window_high))
        bars_since_low = length - window_low.index(min(window_low))
        aroon_up[i] = 100 * (length - bars_since_high) / length
        aroon_down[i] = 100 * (length - bars_since_low) / length
    return aroon_up, aroon_down


def dpo(close: list[float], period: int = 21) -> list[float]:
    """Detrended Price Oscillator (non-centered)."""
    barsback = period // 2 + 1
    ma = [np.nan] * len(close)
    for i in range(period - 1, len(close)):
        ma[i] = sum(close[i-period+1:i+1]) / period
    dpo_values = [np.nan] * len(close)
    for i in range(len(close)):
        if i >= barsback:
            dpo_values[i] = close[i] - ma[i-barsback]
    return dpo_values

def sma(close: list[float], period: int = 100) -> list[float]:
    """Simple Moving Average."""
    sma_values = [np.nan] * len(close)
    for i in range(period - 1, len(close)):
        sma_values[i] = sum(close[i-period+1:i+1]) / period
    return sma_values

def adx(high: list[float], low: list[float], close: list[float], period: int = 14) -> list[float]:
    """Average Directional Index."""
    up_move = [high[i] - high[i - 1] if i > 0 else 0 for i in range(len(high))]
    down_move = [low[i - 1] - low[i] if i > 0 else 0 for i in range(len(low))]
    plus_dm = [up_move[i] if up_move[i] > down_move[i] and up_move[i] > 0 else 0 for i in range(len(up_move))]
    minus_dm = [down_move[i] if down_move[i] > up_move[i] and down_move[i] > 0 else 0 for i in range(len(down_move))]

    atr_values = atr(high, low, close, period)

    plus_di = [100 * (sum(plus_dm[i-period+1:i+1]) / period) / atr_values[i] if i >= period and atr_values[i] != 0 else np.nan for i in range(len(plus_dm))]
    minus_di = [100 * (sum(minus_dm[i-period+1:i+1]) / period) / atr_values[i] if i >= period and atr_values[i] != 0 else np.nan for i in range(len(minus_dm))]

    dx = [100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if i >= period and (plus_di[i] + minus_di[i]) != 0 else np.nan for i in range(len(plus_di))]

    adx_values = [np.nan] * len(dx)
    for i in range(2 * period - 1, len(dx)):
        adx_values[i] = sum(dx[i-period+1:i+1]) / period
    return adx_values

def atr(high: list[float], low: list[float], close: list[float], period: int = 14) -> list[float]:
    """Average True Range."""
    true_range = [0.0] * len(high)
    for i in range(1, len(high)):
        high_low = high[i] - low[i]
        high_close = abs(high[i] - close[i - 1])
        low_close = abs(low[i] - close[i - 1])
        true_range[i] = max(high_low, high_close, low_close)

    atr_values = [np.nan] * len(high)
    for i in range(period, len(high)):
        atr_values[i] = sum(true_range[i-period:i]) / period
    return atr_values

@register
class DgNnfxAroon(BaseStrategy):
    name = "dg_nnfx_aroon"
    version = "1.0.0"
    description = "DillonGrech NNFX: Aroon trigger + DPO/ADX/SMA baseline filters"
    category = "multi_factor"
    tags = ["aroon", "dpo", "adx", "sma"]

    def __init__(self, aroon_length: int = 14, dpo_period: int = 21, adx_period: int = 14, adx_threshold: float = 15.0, baseline_period: int = 100):
        self.aroon_length = aroon_length
        self.dpo_period = dpo_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.baseline_period = baseline_period

    def get_parameter_schema(self) -> dict:
        return {
            "aroon_length": {"type": "integer", "default": 14, "minimum": 1},
            "dpo_period": {"type": "integer", "default": 21, "minimum": 1},
            "adx_period": {"type": "integer", "default": 14, "minimum": 1},
            "adx_threshold": {"type": "number", "default": 15.0, "minimum": 0.0},
            "baseline_period": {"type": "integer", "default": 100, "minimum": 1},
        }

    def get_warmup_periods(self) -> int:
        return max(self.aroon_length, self.dpo_period, self.adx_period, self.baseline_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        high = df["high"].to_list()
        low = df["low"].to_list()
        close = df["close"].to_list()

        if len(close) < self.get_warmup_periods():
            return None

        aroon_up, aroon_down = aroon(high, low, self.aroon_length)
        dpo_values = dpo(close, self.dpo_period)
        adx_values = adx(high, low, close, self.adx_period)
        baseline_values = sma(close, self.baseline_period)

        au = aroon_up[-1]
        ad = aroon_down[-1]
        dpo_val = dpo_values[-1]
        adx_val = adx_values[-1]
        bl = baseline_values[-1]

        au_prev = aroon_up[-2] if len(aroon_up) > 1 else np.nan
        ad_prev = aroon_down[-2] if len(aroon_down) > 1 else np.nan
        au_prev2 = aroon_up[-3] if len(aroon_up) > 2 else np.nan
        ad_prev2 = aroon_down[-3] if len(aroon_down) > 2 else np.nan
        dpo_prev = dpo_values[-2] if len(dpo_values) > 1 else np.nan
        adx_prev = adx_values[-2] if len(adx_values) > 1 else np.nan
        close_prev = close[-2] if len(close) > 1 else np.nan
        bl_prev = baseline_values[-2] if len(baseline_values) > 1 else np.nan

        # C1 trigger: Aroon up crosses above down
        cross_up = (au_prev > ad_prev) and (au_prev2 <= ad_prev2) if not (math.isnan(au_prev) or math.isnan(ad_prev) or math.isnan(au_prev2) or math.isnan(ad_prev2)) else False
        # Filters
        dpo_ok = dpo_prev > 0 if not math.isnan(dpo_prev) else False
        adx_ok = adx_prev > self.adx_threshold if not math.isnan(adx_prev) else False
        bl_ok = close_prev > bl_prev if not math.isnan(close_prev) or not math.isnan(bl_prev) else False

        if cross_up and dpo_ok and adx_ok and bl_ok:
            return Signal(action="buy", strength=1.0, confidence=1.0)

        # Exit: Aroon down crosses above up
        cross_down = (au_prev < ad_prev) and (au_prev2 >= ad_prev2) if not (math.isnan(au_prev) or math.isnan(ad_prev) or math.isnan(au_prev2) or math.isnan(ad_prev2)) else False

        if cross_down:
            return Signal(action="sell", strength=1.0, confidence=1.0)

        return None