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

def rsi(prices: list[float], period: int = 14) -> list[float]:
    if len(prices) < period + 1:
        return [np.nan] * len(prices)

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    avg_gain = np.mean([d for d in deltas[:period] if d > 0])
    avg_loss = np.mean([-d for d in deltas[:period] if d < 0])

    rsi_values = [np.nan] * period
    if avg_loss == 0:
        rsi_values.append(100.0 if avg_gain > 0 else 0.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs)))

    for i in range(period + 1, len(prices)):
        delta = prices[i] - prices[i - 1]
        gain = delta if delta > 0 else 0
        loss = -delta if delta < 0 else 0

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi_values.append(100.0 if avg_gain > 0 else 0.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))

    return rsi_values

def ema(prices: list[float], period: int = 20) -> list[float]:
    if len(prices) < period:
        return [np.nan] * len(prices)

    ema_values = [np.nan] * period
    sma = sum(prices[:period]) / period
    ema_values.append(sma)

    multiplier = 2 / (period + 1)
    for i in range(period, len(prices)):
        ema_values.append((prices[i] - ema_values[-1]) * multiplier + ema_values[-1])

    return ema_values

def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> list[float]:
    if len(highs) < period or len(lows) < period or len(closes) < period:
        return [np.nan] * len(highs)

    tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
          if i > 0 else highs[i] - lows[i] for i in range(len(highs))]

    atr_values = [np.nan] * period
    atr_values.append(sum(tr[1:period+1]) / period)

    for i in range(period + 1, len(highs)):
        atr_values.append((atr_values[-1] * (period - 1) + tr[i]) / period)

    return atr_values

def volume_sma(volumes: list[float], period: int = 20) -> list[float]:
    if len(volumes) < period:
        return [np.nan] * len(volumes)

    sma_values = [np.nan] * period
    for i in range(period, len(volumes)):
        sma_values.append(sum(volumes[i-period:i]) / period)
    return sma_values

@register
class MomentumDayTrading(BaseStrategy):
    name = "momentum_day_trading"
    version = "1.0.0"
    description = "Momentum day trading: buy on combined price + volume spikes"
    category = "momentum"
    tags = ["momentum", "day_trading"]

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_overbought: float = 75.0,
        price_move_pct: float = 0.02,
        vol_mult: float = 2.0,
        vol_sma_period: int = 20,
        ema_period: int = 20,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        max_hold: int = 5,
    ):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.price_move_pct = price_move_pct
        self.vol_mult = vol_mult
        self.vol_sma_period = vol_sma_period
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.max_hold = max_hold
        self.trades = []
        self.in_trade = False
        self.trade_entry_price = 0.0
        self.hold_count = 0
        self.highest_price = 0.0

    def get_parameter_schema(self) -> dict:
        return {
            "rsi_period": {"type": "integer", "default": 14},
            "rsi_overbought": {"type": "number", "default": 75.0},
            "price_move_pct": {"type": "number", "default": 0.02},
            "vol_mult": {"type": "number", "default": 2.0},
            "vol_sma_period": {"type": "integer", "default": 20},
            "ema_period": {"type": "integer", "default": 20},
            "atr_period": {"type": "integer", "default": 14},
            "atr_stop_mult": {"type": "number", "default": 2.0},
            "max_hold": {"type": "integer", "default": 5},
        }

    def get_warmup_periods(self) -> int:
        return max(self.rsi_period, self.vol_sma_period, self.ema_period, self.atr_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        volumes = df["volume"].to_list()

        if len(closes) < max(self.rsi_period, self.vol_sma_period, self.ema_period, self.atr_period) + 1:
            return None

        rsi_values = rsi(closes, period=self.rsi_period)
        ema_values = ema(closes, period=self.ema_period)
        atr_values = atr(highs, lows, closes, period=self.atr_period)
        vol_sma_values = volume_sma(volumes, period=self.vol_sma_period)

        rsi_val = rsi_values[-1]
        ema_val = ema_values[-1]
        atr_val = atr_values[-1]
        vol_sma_val = vol_sma_values[-1]

        close = closes[-1]
        high = highs[-1]
        low = lows[-1]
        volume = volumes[-1]

        if len(closes) < 2:
            return None

        prev_close = closes[-2]
        prev_volume = volumes[-2]

        if math.isnan(vol_sma_val) or math.isnan(ema_val):
            return None

        price_spike = (close - prev_close) / prev_close > self.price_move_pct
        volume_spike = prev_volume > (self.vol_mult * vol_sma_val)
        uptrend = prev_close > ema_val

        if price_spike and volume_spike and uptrend:
            return Signal(action="buy", strength=1.0, confidence=1.0)

        if self.in_trade:
            self.hold_count += 1
            self.highest_price = max(self.highest_price, high)
            trailing_stop = self.highest_price - self.atr_stop_mult * atr_val
            stop_hit = low <= trailing_stop and not math.isnan(atr_val) and atr_val > 0
            rsi_exit = not math.isnan(rsi_val) and rsi_val > self.rsi_overbought
            max_hold_reached = self.hold_count > self.max_hold

            if stop_hit or rsi_exit or max_hold_reached:
                self.in_trade = False
                self.trade_entry_price = 0.0
                self.hold_count = 0
                self.highest_price = 0.0
                return Signal(action="sell", strength=1.0, confidence=1.0)

        return None