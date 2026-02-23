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
    tr = [max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])) if i > 0 else high[i] - low[i] for i in range(len(high))]
    atr_values = []
    for i in range(len(tr)):
        if i < period:
            atr_values.append(np.nan)
        elif i == period:
            atr_values.append(sum(tr[i-period:i]) / period)
        else:
            atr_values.append((atr_values[-1] * (period - 1) + tr[i]) / period)
    return atr_values

def rsi(close: list[float], period: int) -> list[float]:
    deltas = [close[i] - close[i-1] if i > 0 else 0 for i in range(len(close))]
    avg_gain = []
    avg_loss = []
    rsi_values = []

    for i in range(len(close)):
        if i < period:
            avg_gain.append(np.nan)
            avg_loss.append(np.nan)
            rsi_values.append(np.nan)
        elif i == period:
            gains = [d for d in deltas[1:i+1] if d > 0]
            losses = [-d for d in deltas[1:i+1] if d < 0]
            avg_gain.append(sum(gains) / period if gains else 0)
            avg_loss.append(sum(losses) / period if losses else 0)
            if avg_loss[-1] == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain[-1] / avg_loss[-1]
                rsi_values.append(100 - (100 / (1 + rs)))
        else:
            gain = deltas[i] if deltas[i] > 0 else 0
            loss = -deltas[i] if deltas[i] < 0 else 0
            avg_gain.append((avg_gain[-1] * (period - 1) + gain) / period)
            avg_loss.append((avg_loss[-1] * (period - 1) + loss) / period)
            if avg_loss[-1] == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain[-1] / avg_loss[-1]
                rsi_values.append(100 - (100 / (1 + rs)))
    return rsi_values

def macd(close: list[float], fast: int, slow: int, signal: int) -> tuple[list[float], list[float], list[float]]:
    def ema(values: list[float], period: int) -> list[float]:
        ema_values = [np.nan] * len(values)
        if len(values) < period:
            return ema_values
        
        sma = sum(values[:period]) / period
        ema_values[period - 1] = sma
        multiplier = 2 / (period + 1)
        for i in range(period, len(values)):
            ema_values[i] = (values[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        return ema_values

    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = [ema_fast[i] - ema_slow[i] if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]) else np.nan for i in range(len(close))]
    signal_line = ema(macd_line, signal)
    histogram = [macd_line[i] - signal_line[i] if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]) else np.nan for i in range(len(close))]
    return macd_line, signal_line, histogram

def bollinger_bands(close: list[float], period: int, std: float) -> tuple[list[float], list[float], list[float]]:
    upper_band = [np.nan] * len(close)
    middle_band = [np.nan] * len(close)
    lower_band = [np.nan] * len(close)

    for i in range(period - 1, len(close)):
        window = close[i - period + 1:i + 1]
        mean = sum(window) / period
        std_dev = math.sqrt(sum([(x - mean) ** 2 for x in window]) / period)
        middle_band[i] = mean
        upper_band[i] = mean + std * std_dev
        lower_band[i] = mean - std * std_dev

    return upper_band, middle_band, lower_band

def directional_indicators(high: list[float], low: list[float], close: list[float], period: int) -> tuple[list[float], list[float], list[float]]:
    plus_dm = [0.0] * len(high)
    minus_dm = [0.0] * len(high)
    tr = [0.0] * len(high)
    
    for i in range(1, len(high)):
        move_up = high[i] - high[i-1]
        move_down = low[i-1] - low[i]
        
        plus_dm[i] = move_up if move_up > move_down and move_up > 0 else 0.0
        minus_dm[i] = move_down if move_down > move_up and move_down > 0 else 0.0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
    plus_dm_smooth = [np.nan] * len(high)
    minus_dm_smooth = [np.nan] * len(high)
    tr_smooth = [np.nan] * len(high)
    
    for i in range(period, len(high)):
        plus_dm_smooth[i] = sum(plus_dm[i-period+1:i+1]) / period
        minus_dm_smooth[i] = sum(minus_dm[i-period+1:i+1]) / period
        tr_smooth[i] = sum(tr[i-period+1:i+1]) / period
        
    plus_di = [np.nan] * len(high)
    minus_di = [np.nan] * len(high)
    
    for i in range(period, len(high)):
        if tr_smooth[i] != 0:
            plus_di[i] = 100 * (plus_dm_smooth[i] / tr_smooth[i])
            minus_di[i] = 100 * (minus_dm_smooth[i] / tr_smooth[i])
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
            
    dx = [np.nan] * len(high)
    for i in range(period, len(high)):
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0.0
            
    adx = [np.nan] * len(high)
    adx_smooth = [np.nan] * len(high)
    
    for i in range(2*period, len(high)):
        adx_smooth[i] = sum(dx[i-period+1:i+1]) / period
        
    adx = adx_smooth
    
    return adx, plus_di, minus_di

@register
class PullbackTrendContinuation(BaseStrategy):
    name = "pullback_trend_continuation"
    version = "1.0.0"
    description = "Pullback-to-trend continuation: buy pullbacks in confirmed uptrends"
    category = "trend"
    tags = ["pullback", "trend", "continuation"]

    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        di_ratio: float = 1.5,
        rsi_low: float = 50.0,
        rsi_high: float = 65.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
        stop_atr_mult: float = 2.0,
        max_hold: int = 15,
    ):
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.di_ratio = di_ratio
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        self.max_hold = max_hold

    def get_warmup_periods(self) -> int:
        return max(self.adx_period, self.bb_period, self.macd_slow, self.atr_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        opens = df["open"].to_list()
        volumes = df["volume"].to_list()

        if len(closes) < self.get_warmup_periods():
            return None

        adx, plus_di, minus_di = directional_indicators(highs, lows, closes, self.adx_period)
        macd_line, signal_line, histogram = macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        rsi_values = rsi(closes, 14)
        upper_band, middle_band, lower_band = bollinger_bands(closes, self.bb_period, self.bb_std)
        atr_values = atr(highs, lows, closes, self.atr_period)

        adx_val = adx[-2]
        plus_di_val = plus_di[-2]
        minus_di_val = minus_di[-2]
        macd_line_val = macd_line[-2]
        signal_line_val = signal_line[-2]
        histogram_val = histogram[-2]
        rsi_val = rsi_values[-2]
        close_val = closes[-2]
        open_val = opens[-2]
        middle_band_val = middle_band[-2]

        if (
            (adx_val is not None and adx_val > self.adx_threshold) and
            (plus_di_val is not None and minus_di_val is not None and plus_di_val > minus_di_val * self.di_ratio) and
            (macd_line_val is not None and signal_line_val is not None and macd_line_val > signal_line_val) and
            (rsi_val is not None and self.rsi_low <= rsi_val <= self.rsi_high) and
            (close_val is not None and middle_band_val is not None and close_val <= middle_band_val * 1.03) and
            (close_val is not None and open_val is not None and close_val > open_val) and
            (histogram_val is not None and histogram_val > 0)
        ):
            return Signal(action="buy", strength=1.0, confidence=1.0)

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "adx_period": {"type": "integer", "default": 14},
                "adx_threshold": {"type": "number", "default": 25.0},
                "di_ratio": {"type": "number", "default": 1.5},
                "rsi_low": {"type": "number", "default": 50.0},
                "rsi_high": {"type": "number", "default": 65.0},
                "bb_period": {"type": "integer", "default": 20},
                "bb_std": {"type": "number", "default": 2.0},
                "macd_fast": {"type": "integer", "default": 12},
                "macd_slow": {"type": "integer", "default": 26},
                "macd_signal": {"type": "integer", "default": 9},
                "atr_period": {"type": "integer", "default": 14},
                "stop_atr_mult": {"type": "number", "default": 2.0},
                "max_hold": {"type": "integer", "default": 15},
            },
        }