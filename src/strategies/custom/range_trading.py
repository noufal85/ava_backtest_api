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

def atr(high: list[float], low: list[float], close: list[float], period: int) -> list[float]:
    tr = [0.0]  # Initialize with a default value
    for i in range(1, len(high)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
    atr_values = []
    for i in range(len(high)):
        if i < period:
            atr_values.append(np.nan)
        else:
            atr_values.append(sum(tr[i - period + 1:i + 1]) / period)
    return atr_values

def rsi(close: list[float], period: int) -> list[float]:
    deltas = [close[i] - close[i - 1] for i in range(1, len(close))]
    avg_gain = []
    avg_loss = []
    for i in range(len(close)):
        if i < period:
            avg_gain.append(np.nan)
            avg_loss.append(np.nan)
        else:
            gains = [d for d in deltas[i - period:i] if d > 0]
            losses = [-d for d in deltas[i - period:i] if d < 0]
            avg_gain.append(sum(gains) / period if gains else 0)
            avg_loss.append(sum(losses) / period if losses else 0)

    rsi_values = []
    for i in range(len(close)):
        if i < period:
            rsi_values.append(np.nan)
        else:
            if avg_loss[i] == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi_values.append(100 - (100 / (1 + rs)))
    return rsi_values

def bollinger_bands(close: list[float], period: int, std: float) -> tuple[list[float], list[float], list[float]]:
    sma_values = []
    for i in range(len(close)):
        if i < period - 1:
            sma_values.append(np.nan)
        else:
            sma_values.append(sum(close[i - period + 1:i + 1]) / period)

    upper_band = []
    lower_band = []
    for i in range(len(close)):
        if i < period - 1:
            upper_band.append(np.nan)
            lower_band.append(np.nan)
        else:
            std_dev = np.std(close[i - period + 1:i + 1])
            upper_band.append(sma_values[i] + std * std_dev)
            lower_band.append(sma_values[i] - std * std_dev)

    return upper_band, lower_band, sma_values

def sma(close: list[float], period: int) -> list[float]:
    sma_values = []
    for i in range(len(close)):
        if i < period - 1:
            sma_values.append(np.nan)
        else:
            sma_values.append(sum(close[i - period + 1:i + 1]) / period)
    return sma_values

@register
class RangeTrading(BaseStrategy):
    name = "range_trading"
    version = "1.0.0"
    description = "Range trading: buy near support, sell near resistance in sideways markets"
    category = "multi_factor"
    tags = ["range", "mean_reversion"]

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        lookback: int = 20,
        atr_period: int = 14,
        bandwidth_max: float = 0.15,
        max_hold: int = 10,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.lookback = lookback
        self.atr_period = atr_period
        self.bandwidth_max = bandwidth_max
        self.max_hold = max_hold
        self.hold_count = 0
        self.in_trade = False
        self.entry_price = 0.0
        self.atr_value = 0.0

    def get_warmup_periods(self) -> int:
        return max(self.bb_period, self.rsi_period, self.lookback, self.atr_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()

        if len(closes) < self.get_warmup_periods():
            return None

        bb_upper, bb_lower, bb_middle = bollinger_bands(closes, self.bb_period, self.bb_std)
        rsi_values = rsi(closes, self.rsi_period)
        atr_values = atr(highs, lows, closes, self.atr_period)

        bandwidth = [(bb_upper[i] - bb_lower[i]) / bb_middle[i] if bb_middle[i] != 0 else np.nan for i in range(len(closes))]

        support = []
        resistance = []
        for i in range(len(closes)):
            if i < self.lookback:
                support.append(np.nan)
                resistance.append(np.nan)
            else:
                support.append(min(lows[i-self.lookback:i]))
                resistance.append(max(highs[i-self.lookback:i]))

        current_close = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else np.nan
        prev_rsi = rsi_values[-2] if len(rsi_values) > 1 else np.nan
        prev_bb_lower = bb_lower[-2] if len(bb_lower) > 1 else np.nan
        prev_bandwidth = bandwidth[-2] if len(bandwidth) > 1 else np.nan
        prev_support = support[-2] if len(support) > 1 else np.nan
        current_atr = atr_values[-1]

        in_range = prev_bandwidth < self.bandwidth_max if not np.isnan(prev_bandwidth) else False
        near_support = (prev_close <= prev_bb_lower * 1.02 if not np.isnan(prev_bb_lower) else False) or \
                       (prev_close <= prev_support * 1.02 if not np.isnan(prev_support) else False)
        oversold = prev_rsi < self.rsi_oversold if not np.isnan(prev_rsi) else False

        near_resistance = (current_close >= bb_upper[-1] * 0.98 if not np.isnan(bb_upper[-1]) else False)
        rsi_high = rsi_values[-1] > self.rsi_overbought if not np.isnan(rsi_values[-1]) else False
        stop_price = self.entry_price - 1.5 * self.atr_value
        stop_hit = lows[-1] <= stop_price and self.atr_value > 0

        if not self.in_trade:
            if in_range and near_support and oversold:
                self.in_trade = True
                self.hold_count = 1
                self.entry_price = current_close
                self.atr_value = current_atr
                return Signal(action="buy", strength=1.0, confidence=0.8)
            else:
                return None
        else:
            self.hold_count += 1
            max_hold_reached = self.hold_count > self.max_hold

            if near_resistance or rsi_high or stop_hit or max_hold_reached:
                self.in_trade = False
                self.hold_count = 0
                self.entry_price = 0.0
                self.atr_value = 0.0
                return Signal(action="sell", strength=1.0, confidence=0.8)
            else:
                return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "bb_period": {"type": "integer", "default": 20},
                "bb_std": {"type": "number", "default": 2.0},
                "rsi_period": {"type": "integer", "default": 14},
                "rsi_oversold": {"type": "number", "default": 35.0},
                "rsi_overbought": {"type": "number", "default": 65.0},
                "lookback": {"type": "integer", "default": 20},
                "atr_period": {"type": "integer", "default": 14},
                "bandwidth_max": {"type": "number", "default": 0.15},
                "max_hold": {"type": "integer", "default": 10},
            },
        }