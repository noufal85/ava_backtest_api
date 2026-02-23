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

def _lwti(close: list[float], period: int = 25) -> list[float]:
    """Larry Williams Large Trade Index."""
    lwti = []
    for i in range(len(close)):
        if i < period:
            lwti.append(np.nan)
            continue
        
        ma = sum([close[i] - close[i-j] for j in range(1, period+1)]) / period
        
        tr_values = [abs(close[i-j] - close[i-j-1]) if i-j-1 >= 0 else 0 for j in range(0, period)]
        atr_val = tr_values[0]
        for j in range(1, len(tr_values)):
            atr_val = (tr_values[j] + (period - 1) * atr_val) / period
        
        lwti_val = ma / atr_val * 50 + 50
        lwti.append(lwti_val)
    return lwti

@register
class DgDonchianLwtiVol(BaseStrategy):
    name = "dg_donchian_lwti_vol"
    version = "1.0.0"
    description = "DillonGrech Donchian+LWTI+Volume: breakout with momentum and volume filters"
    category = "momentum"
    tags = ["donchian", "lwti", "volume", "breakout"]

    def __init__(self, don_length: int = 96, lwti_period: int = 25, vol_ma_period: int = 30, profit_rr: float = 2.0):
        self.don_length = don_length
        self.lwti_period = lwti_period
        self.vol_ma_period = vol_ma_period
        self.profit_rr = profit_rr
        self.trade_taken = False
        self.stop_price = None
        self.tp_price = None
        self.basis = None

    def get_parameter_schema(self) -> dict:
        return {
            "don_length": {"type": "integer", "default": 96, "minimum": 1},
            "lwti_period": {"type": "integer", "default": 25, "minimum": 1},
            "vol_ma_period": {"type": "integer", "default": 30, "minimum": 1},
            "profit_rr": {"type": "number", "default": 2.0, "minimum": 0.1},
        }

    def get_warmup_periods(self) -> int:
        return max(self.don_length, self.lwti_period, self.vol_ma_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical, current_bar])

        if len(df) < self.get_warmup_periods():
            return None

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        volumes = df["volume"].to_list()

        don_upper = [max(highs[i - self.don_length:i]) if i >= self.don_length else np.nan for i in range(len(highs))]
        don_lower = [min(lows[i - self.don_length:i]) if i >= self.don_length else np.nan for i in range(len(lows))]
        don_basis = [(don_upper[i] + don_lower[i]) / 2 if not np.isnan(don_upper[i]) and not np.isnan(don_lower[i]) else np.nan for i in range(len(don_upper))]
        lwti = _lwti(closes, self.lwti_period)
        vol_ma = [sum(volumes[i - self.vol_ma_period:i]) / self.vol_ma_period if i >= self.vol_ma_period else np.nan for i in range(len(volumes))]

        current_idx = len(df) - 1

        if current_idx < 2:
            return None

        prev_close = closes[current_idx - 1]
        prev2_close = closes[current_idx - 2]
        prev_don_upper = don_upper[current_idx - 2]
        prev_lwti = lwti[current_idx - 1]
        prev_vol = volumes[current_idx - 1]
        prev_vol_ma = vol_ma[current_idx - 1]
        prev_don_basis = don_basis[current_idx - 1]
        price = current_bar["open"][0]
        self.basis = don_basis[current_idx]

        # Reset trade counter on basis cross
        if current_idx > 0:
            if not np.isnan(self.basis) and not np.isnan(prev_close):
                if (prev_close > self.basis and price < self.basis) or (prev_close < self.basis and price > self.basis):
                    self.trade_taken = False

        if self.trade_taken:
            low, high = current_bar["low"][0], current_bar["high"][0]
            if not np.isnan(low) and self.stop_price is not None and low <= self.stop_price:
                self.trade_taken = False
                return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"reason": "stop_loss"})
            if not np.isnan(high) and self.tp_price is not None and high >= self.tp_price:
                self.trade_taken = False
                return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"reason": "take_profit"})
            if prev_close < prev_don_basis:
                self.trade_taken = False
                return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"reason": "basis_cross"})
            return None

        buy = (prev_close > prev_don_upper) and (prev2_close <= prev_don_upper) and \
              (prev_lwti > 50) and (prev_vol > prev_vol_ma)

        if buy and not self.trade_taken:
            if not np.isnan(self.basis):
                stop_dist = abs(price - self.basis) * 1.02
                if stop_dist < price * 0.001:
                    stop_dist = price * 0.02
                self.stop_price = price - stop_dist
                self.tp_price = price + stop_dist * self.profit_rr
                self.trade_taken = True
                return Signal(action="buy", strength=1.0, confidence=1.0)

        return None