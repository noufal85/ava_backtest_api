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

def rsi(closes: list[float], period: int) -> list[float]:
    """Calculates the Relative Strength Index (RSI)."""
    if len(closes) < period + 1:
        return [np.nan] * len(closes)

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    avg_gain = sum(d for d in deltas[:period] if d > 0) / period
    avg_loss = abs(sum(d for d in deltas[:period] if d < 0) / period)

    rsi_values = [np.nan] * period
    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs)))

    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain = delta if delta > 0 else 0
        loss = abs(delta) if delta < 0 else 0

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))

    return rsi_values

def ema(series: list[float], period: int) -> list[float]:
    """Calculates the Exponential Moving Average (EMA)."""
    if len(series) < period:
        return [np.nan] * len(series)

    ema_values = [np.nan] * (period - 1)
    sma = sum(series[:period]) / period
    ema_values.append(sma)

    k = 2 / (period + 1)
    for i in range(period, len(series)):
        ema_values.append((series[i] * k) + (ema_values[-1] * (1 - k)))

    return ema_values

def macd(closes: list[float], fast: int, slow: int, signal: int) -> tuple[list[float], list[float], list[float]]:
    """Calculates the Moving Average Convergence Divergence (MACD)."""
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)

    macd_line = [
        ema_fast[i] - ema_slow[i]
        if not math.isnan(ema_fast[i]) and not math.isnan(ema_slow[i])
        else np.nan
        for i in range(len(closes))
    ]
    signal_line = ema(macd_line, signal)
    histogram = [
        macd_line[i] - signal_line[i]
        if not math.isnan(macd_line[i]) and not math.isnan(signal_line[i])
        else np.nan
        for i in range(len(closes))
    ]

    return macd_line, signal_line, histogram

@register
class RsiDivergenceMacd(BaseStrategy):
    name = "rsi_divergence_macd"
    version = "1.1.0"
    description = "RSI divergence (bullish & bearish) confirmed by MACD histogram crossover"
    category = "mean_reversion"
    tags = ["rsi", "macd", "divergence"]

    def __init__(
        self,
        rsi_period: int = 14,
        divergence_window: int = 10,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ):
        self.rsi_period = rsi_period
        self.divergence_window = divergence_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def get_parameter_schema(self) -> dict:
        return {
            "rsi_period": {"type": "integer", "default": 14, "minimum": 2, "maximum": 50},
            "divergence_window": {"type": "integer", "default": 10, "minimum": 2, "maximum": 50},
            "macd_fast": {"type": "integer", "default": 12, "minimum": 2, "maximum": 50},
            "macd_slow": {"type": "integer", "default": 26, "minimum": 2, "maximum": 100},
            "macd_signal": {"type": "integer", "default": 9, "minimum": 2, "maximum": 50},
        }

    def get_warmup_periods(self) -> int:
        return max(self.rsi_period, self.divergence_window, self.macd_slow) + 5

    def generate_signal(self, window) -> Signal | None:
        df = pl.concat([window.historical(), window.current_bar()])
        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()

        if len(closes) < self.get_warmup_periods():
            return None

        rsi_values = rsi(closes, self.rsi_period)
        macd_line, signal_line, macd_hist = macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)

        rsi_val = rsi_values[-1]
        macd_hist_val = macd_hist[-1]
        macd_hist_val_prev = macd_hist[-2]
        close_val = closes[-1]

        # === Bullish divergence: price lower low, RSI higher low ===
        window_closes = closes[-(self.divergence_window + 1):-1]
        window_rsi = rsi_values[-(self.divergence_window + 1):-1]
        price_min = min(window_closes)
        rsi_at_price_min_window = min(window_rsi)

        price_near_low = closes[-2] <= price_min * 1.01
        rsi_higher = rsi_values[-2] > rsi_at_price_min_window + 2
        macd_cross_up = (macd_hist_val_prev > 0) and (macd_hist[-3] <= 0 if len(macd_hist) >= 3 else False)
        if price_near_low and rsi_higher and macd_cross_up:
            return Signal(action="buy", strength=min(rsi_val / 70, 1.0), confidence=0.7)

        # === Bearish divergence: price higher high, RSI lower high ===
        window_closes = closes[-(self.divergence_window + 1):-1]
        window_rsi = rsi_values[-(self.divergence_window + 1):-1]
        price_max = max(window_closes)
        rsi_at_price_max_window = max(window_rsi)

        price_near_high = closes[-2] >= price_max * 0.99
        rsi_lower = rsi_values[-2] < rsi_at_price_max_window - 2
        macd_cross_down = (macd_hist_val_prev < 0) and (macd_hist[-3] >= 0 if len(macd_hist) >= 3 else False)

        if price_near_high and rsi_lower and macd_cross_down:
            return Signal(action="sell", strength=min((100 - rsi_val) / 70, 1.0), confidence=0.7)

        # === Exit signals ===
        rsi_prev = rsi_values[-2]
        if rsi_prev > 70 or (macd_hist_val_prev < 0 and (macd_hist[-3] >= 0 if len(macd_hist) >= 3 else False)):
            return Signal(action="hold", strength=0.0, confidence=0.9, metadata={"exit_reason": "rsi_overbought_or_macd_bearish"})

        if rsi_prev < 30 or (macd_hist_val_prev > 0 and (macd_hist[-3] <= 0 if len(macd_hist) >= 3 else False)):
            return Signal(action="hold", strength=0.0, confidence=0.9, metadata={"exit_reason": "rsi_oversold_or_macd_bullish"})

        return None