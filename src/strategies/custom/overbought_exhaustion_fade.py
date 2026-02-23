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
    """Calculates the Relative Strength Index (RSI) for a given period."""
    if len(closes) < period + 1:
        return [np.nan] * len(closes)

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    avg_gain = np.mean([d for d in deltas[:period] if d > 0])
    avg_loss = -np.mean([d for d in deltas[:period] if d < 0])
    rsi_values = [np.nan] * period

    for i in range(period, len(closes) - 1):
        up = deltas[i] if deltas[i] > 0 else 0
        down = -deltas[i] if deltas[i] < 0 else 0
        avg_gain = (avg_gain * (period - 1) + up) / period
        avg_loss = (avg_loss * (period - 1) + down) / period
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))

    return rsi_values + [np.nan]

def macd(closes: list[float], fast: int, slow: int, signal: int) -> tuple[list[float], list[float], list[float]]:
    """Calculates MACD, signal line, and histogram."""
    def ema(closes: list[float], period: int) -> list[float]:
        ema = [np.nan] * len(closes)
        if len(closes) < period:
            return ema
        
        k = 2 / (period + 1)
        ema[period-1] = np.mean(closes[:period])
        for i in range(period, len(closes)):
            ema[i] = (closes[i] * k) + (ema[i-1] * (1 - k))
        return ema

    macd_line = [np.nan] * len(closes)
    signal_line = [np.nan] * len(closes)
    histogram = [np.nan] * len(closes)

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)

    for i in range(len(closes)):
        if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]):
            macd_line[i] = ema_fast[i] - ema_slow[i]

    signal_line = ema(macd_line, signal)

    for i in range(len(closes)):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, histogram

def bollinger_bands(closes: list[float], period: int, std: float) -> tuple[list[float], list[float], list[float]]:
    """Calculates Bollinger Bands."""
    upper_band = [np.nan] * len(closes)
    middle_band = [np.nan] * len(closes)
    lower_band = [np.nan] * len(closes)

    if len(closes) < period:
        return upper_band, middle_band, lower_band

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        mean = np.mean(window)
        std_dev = np.std(window)
        middle_band[i] = mean
        upper_band[i] = mean + std * std_dev
        lower_band[i] = mean - std * std_dev

    return upper_band, middle_band, lower_band

def atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float]:
    """Calculates Average True Range (ATR)."""
    true_range = [0.0] * len(highs)
    atr_values = [np.nan] * len(highs)

    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        true_range[i] = max(high_low, high_close, low_close)

    atr_values[period - 1] = np.mean(true_range[1:period+1])

    for i in range(period, len(highs)):
        atr_values[i] = (atr_values[i - 1] * (period - 1) + true_range[i]) / period

    return atr_values

def directional_indicators(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float]:
    """Calculates ADX (Average Directional Index)."""
    
    pdm = [0.0] * len(highs)
    mdm = [0.0] * len(highs)
    tr = [0.0] * len(highs)
    
    for i in range(1, len(highs)):
        pdm[i] = max(0, highs[i] - highs[i-1]) if (highs[i] - highs[i-1]) > (lows[i-1] - lows[i]) else 0
        mdm[i] = max(0, lows[i-1] - lows[i]) if (highs[i] - highs[i-1]) < (lows[i-1] - lows[i]) else 0
        tr_val = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr[i] = tr_val

    atr_pdm = [np.nan] * len(highs)
    atr_mdm = [np.nan] * len(highs)
    atr_tr = [np.nan] * len(highs)

    atr_pdm[period] = np.mean(pdm[1:period+1])
    atr_mdm[period] = np.mean(mdm[1:period+1])
    atr_tr[period] = np.mean(tr[1:period+1])

    for i in range(period + 1, len(highs)):
        atr_pdm[i] = (atr_pdm[i-1] * (period - 1) + pdm[i]) / period
        atr_mdm[i] = (atr_mdm[i-1] * (period - 1) + mdm[i]) / period
        atr_tr[i] = (atr_tr[i-1] * (period - 1) + tr[i]) / period

    di_plus = [np.nan] * len(highs)
    di_minus = [np.nan] * len(highs)
    for i in range(len(highs)):
        if atr_tr[i] != 0:
            di_plus[i] = 100 * (atr_pdm[i] / atr_tr[i])
            di_minus[i] = 100 * (atr_mdm[i] / atr_tr[i])

    dx = [np.nan] * len(highs)
    for i in range(len(highs)):
        if (di_plus[i] + di_minus[i]) != 0:
            dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])

    adx = [np.nan] * len(highs)
    adx[2 * period - 1] = np.mean(dx[period:2*period])

    for i in range(2 * period, len(highs)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    return adx

@register
class OverboughtExhaustionFade(BaseStrategy):
    name = "overbought_exhaustion_fade"
    version = "1.0.0"
    description = "Counter-trend short on overbought exhaustion with momentum fade"
    category = "mean_reversion"
    tags = []

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_overbought: float = 68.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        histogram_decline_bars: int = 2,
        atr_period: int = 14,
        stop_atr_mult: float = 1.5,
        max_hold: int = 10,
    ):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.histogram_decline_bars = histogram_decline_bars
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        self.max_hold = max_hold

    def get_warmup_periods(self) -> int:
        return max(self.rsi_period, self.bb_period, self.macd_fast, self.macd_slow, self.macd_signal, self.atr_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical, current_bar])

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        opens = df["open"].to_list()
        
        if len(closes) < self.get_warmup_periods():
            return None

        adx = directional_indicators(highs, lows, closes, self.atr_period)[-1]
        macd_line, signal_line, histogram = macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        rsi_val = rsi(closes, self.rsi_period)[-1]
        upper_band, middle_band, lower_band = bollinger_bands(closes, self.bb_period, self.bb_std)
        atr_val = atr(highs, lows, closes, self.atr_period)[-1]

        # RSI overbought
        cond_rsi = rsi_val > self.rsi_overbought

        # MACD histogram declining for N bars
        hist_declining = True
        if len(histogram) >= self.histogram_decline_bars + 1:
            for j in range(1, self.histogram_decline_bars + 1):
                if histogram[-j-1] <= histogram[-j]:
                    hist_declining = False
                    break
        else:
            hist_declining = False

        # Histogram was > 1.0 within 5 bars but now < 1.0
        hist_was_high = False
        if len(histogram) > 6:
            for j in range(1, 6):
                if histogram[-j-1] > 1.0:
                    hist_was_high = True
                    break
        hist_now_low = histogram[-2] < 1.0 if len(histogram) > 1 else False
        cond_exhaustion = hist_was_high and hist_now_low

        # Price near or above upper BB
        cond_bb = closes[-2] >= upper_band[-2] * 0.98 if len(upper_band) > 1 else False

        # Upper wick > body (rejection)
        body = abs(closes[-2] - opens[-2]) if len(closes) > 1 and len(opens) > 1 else 0
        upper_wick = highs[-2] - max(closes[-2], opens[-2]) if len(highs) > 1 and len(opens) > 1 and len(closes) > 1 else 0
        cond_wick = upper_wick > body

        # ADX > 20
        cond_adx = adx > 20 if adx is not None and not math.isnan(adx) else False

        if cond_rsi and hist_declining and cond_exhaustion and cond_bb and cond_wick and cond_adx:
            return Signal(action="sell", strength=1.0, confidence=1.0)

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "rsi_period": {"type": "integer", "default": 14},
                "rsi_overbought": {"type": "number", "default": 68.0},
                "bb_period": {"type": "integer", "default": 20},
                "bb_std": {"type": "number", "default": 2.0},
                "macd_fast": {"type": "integer", "default": 12},
                "macd_slow": {"type": "integer", "default": 26},
                "macd_signal": {"type": "integer", "default": 9},
                "histogram_decline_bars": {"type": "integer", "default": 2},
                "atr_period": {"type": "integer", "default": 14},
                "stop_atr_mult": {"type": "number", "default": 1.5},
                "max_hold": {"type": "integer", "default": 10},
            },
        }