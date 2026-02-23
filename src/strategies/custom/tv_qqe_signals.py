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
    
    seed_up = sum(d for d in deltas[:period] if d > 0) / period
    seed_down = -sum(d for d in deltas[:period] if d < 0) / period
    rs = seed_up / seed_down if seed_down != 0 else 0
    rsi_first = 100 - 100 / (1 + rs)
    
    rsi_values = [np.nan] * period + [rsi_first]
    
    up = seed_up
    down = seed_down
    
    for i in range(period, len(deltas)):
        delta = deltas[i]
        
        up = (up * (period - 1) + (delta if delta > 0 else 0)) / period
        down = (down * (period - 1) - (delta if delta < 0 else 0)) / period
        
        rs = up / down if down != 0 else 0
        rsi_values.append(100 - 100 / (1 + rs))
        
    return rsi_values

def ema(data: list[float], period: int) -> list[float]:
    """Calculates the Exponential Moving Average (EMA) for a given period."""
    if len(data) < period:
        return [np.nan] * len(data)

    ema = [np.nan] * len(data)
    
    # Simple Moving Average as initial value
    ema[period - 1] = sum(data[:period]) / period
    
    alpha = 2 / (period + 1)
    
    for i in range(period, len(data)):
        ema[i] = (data[i] * alpha) + (ema[i - 1] * (1 - alpha))
        
    return ema

@register
class TvQqeSignals(BaseStrategy):
    name = "tv_qqe_signals"
    version = "1.0.0"
    description = "QQE Signals: Quantitative Qualitative Estimation momentum strategy"
    category = "multi_factor"
    tags = ["momentum", "qqe", "rsi"]

    def __init__(self, rsi_period: int = 14, rsi_smoothing: int = 5, qqe_factor: float = 4.238, threshold: int = 10):
        self.rsi_period = rsi_period
        self.rsi_smoothing = rsi_smoothing
        self.qqe_factor = qqe_factor
        self.threshold = threshold

    def get_parameter_schema(self) -> dict:
        return {
            "rsi_period": {"type": "integer", "default": 14},
            "rsi_smoothing": {"type": "integer", "default": 5},
            "qqe_factor": {"type": "number", "default": 4.238},
            "threshold": {"type": "integer", "default": 10},
        }

    def get_warmup_periods(self) -> int:
        return max(self.rsi_period, self.rsi_smoothing) * 3 # Added buffer

    def generate_signal(self, window) -> Signal | None:
        df = window.historical().sort("ts")
        current_bar = window.current_bar()
        if current_bar is not None:
            df = pl.concat([df, current_bar])
        closes = df["close"].to_list()

        if len(closes) < self.get_warmup_periods():
            return None

        # Calculate RSI and smooth it
        rsi_values = rsi(closes, self.rsi_period)
        rsi_ma_values = ema(rsi_values, self.rsi_smoothing)
        
        # Calculate ATR of RSI
        atr_rsi_values = [abs(rsi_ma_values[i] - rsi_ma_values[i-1]) if i > 0 and not math.isnan(rsi_ma_values[i]) and not math.isnan(rsi_ma_values[i-1]) else np.nan for i in range(len(rsi_ma_values))]
        
        # Wilder's period for smoothing
        wilders_period = self.rsi_period * 2 - 1
        ma_atr_rsi_values = ema(atr_rsi_values, wilders_period)
        dar_values = [val * self.qqe_factor if not math.isnan(val) else np.nan for val in ma_atr_rsi_values]
        
        # QQE bands calculation
        new_longband_values = [rsi_ma_values[i] - dar_values[i] if not math.isnan(rsi_ma_values[i]) and not math.isnan(dar_values[i]) else np.nan for i in range(len(rsi_ma_values))]
        new_shortband_values = [rsi_ma_values[i] + dar_values[i] if not math.isnan(rsi_ma_values[i]) and not math.isnan(dar_values[i]) else np.nan for i in range(len(rsi_ma_values))]
        
        # Initialize bands
        longband_values = [np.nan] * len(rsi_ma_values)
        shortband_values = [np.nan] * len(rsi_ma_values)
        trend_values = [0] * len(rsi_ma_values)
        
        # Calculate dynamic bands
        for i in range(1, len(rsi_ma_values)):
            prev_rsi = rsi_ma_values[i-1] if i > 0 else np.nan
            curr_rsi = rsi_ma_values[i]
            prev_longband = longband_values[i-1] if i > 0 else np.nan
            prev_shortband = shortband_values[i-1] if i > 0 else np.nan
            new_longband = new_longband_values[i]
            new_shortband = new_shortband_values[i]
            
            # Longband logic
            if not math.isnan(prev_rsi) and not math.isnan(prev_longband) and not math.isnan(curr_rsi) and not math.isnan(new_longband):
                if prev_rsi > prev_longband and curr_rsi > prev_longband:
                    longband_values[i] = max(prev_longband, new_longband)
                else:
                    longband_values[i] = new_longband
            
            # Shortband logic
            if not math.isnan(prev_rsi) and not math.isnan(prev_shortband) and not math.isnan(curr_rsi) and not math.isnan(new_shortband):
                if prev_rsi < prev_shortband and curr_rsi < prev_shortband:
                    shortband_values[i] = min(prev_shortband, new_shortband)
                else:
                    shortband_values[i] = new_shortband
        
        # Trend detection
        for i in range(1, len(rsi_ma_values)):
            prev_trend = trend_values[i-1] if i > 0 else 1
            curr_rsi = rsi_ma_values[i]
            prev_shortband = shortband_values[i-1] if i > 0 else np.nan
            prev_longband = longband_values[i-1] if i > 0 else np.nan
            
            if not math.isnan(curr_rsi) and not math.isnan(prev_shortband) and not math.isnan(prev_longband):
                if curr_rsi > prev_shortband:  # Cross above shortband
                    trend_values[i] = 1
                elif curr_rsi < prev_longband:  # Cross below longband
                    trend_values[i] = -1
                else:
                    trend_values[i] = prev_trend
                
        # FastAtrRsiTL (the QQE line)
        fast_atr_rsi_tl_values = [longband_values[i] if trend_values[i] == 1 else shortband_values[i] for i in range(len(rsi_ma_values))]
        
        # Track QQE crosses using shifted values to avoid look-ahead bias
        if len(fast_atr_rsi_tl_values) < 2 or len(rsi_ma_values) < 2:
            return None

        prev_fast_atr = fast_atr_rsi_tl_values[-2]
        prev_rsi_ma = rsi_ma_values[-2]
        curr_rsi_ma = rsi_ma_values[-1]
        curr_fast_atr = fast_atr_rsi_tl_values[-1]
        
        # Long signal: RSI MA crosses above FastAtrRsiTL
        qqe_long = (prev_rsi_ma <= prev_fast_atr) and (curr_rsi_ma > curr_fast_atr) if not math.isnan(prev_rsi_ma) and not math.isnan(prev_fast_atr) and not math.isnan(curr_rsi_ma) and not math.isnan(curr_fast_atr) else False
        
        # Short/Exit signal: RSI MA crosses below FastAtrRsiTL
        qqe_short = (prev_rsi_ma >= prev_fast_atr) and (curr_rsi_ma < curr_fast_atr) if not math.isnan(prev_rsi_ma) and not math.isnan(prev_fast_atr) and not math.isnan(curr_rsi_ma) and not math.isnan(curr_fast_atr) else False

        if qqe_long:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"qqe_value": curr_fast_atr, "rsi_ma": curr_rsi_ma})
        elif qqe_short:
            return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"qqe_value": curr_fast_atr, "rsi_ma": curr_rsi_ma})
        else:
            return None