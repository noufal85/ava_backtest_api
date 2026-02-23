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

def bollinger_bands(closes: list[float], period: int, std_dev: float) -> tuple[list[float], list[float], list[float]]:
    """Calculates Bollinger Bands."""
    upper_band = []
    lower_band = []
    middle_band = []
    for i in range(len(closes)):
        if i < period - 1:
            upper_band.append(np.nan)
            lower_band.append(np.nan)
            middle_band.append(np.nan)
        else:
            window = closes[i - period + 1:i + 1]
            mean = sum(window) / period
            squared_differences = [(x - mean) ** 2 for x in window]
            variance = sum(squared_differences) / period
            std_deviation = math.sqrt(variance)
            upper = mean + std_dev * std_deviation
            lower = mean - std_dev * std_deviation
            upper_band.append(upper)
            lower_band.append(lower)
            middle_band.append(mean)
    return upper_band, lower_band, middle_band

def sma(data: list[float], period: int) -> list[float]:
    """Calculates Simple Moving Average."""
    sma_values = []
    for i in range(len(data)):
        if i < period - 1:
            sma_values.append(np.nan)
        else:
            window = data[i - period + 1:i + 1]
            sma = sum(window) / period
            sma_values.append(sma)
    return sma_values

@register
class TvBollingerBandwidth(BaseStrategy):
    name = "tv_bollinger_bandwidth"
    version = "1.0.0"
    description = "Bollinger Bandwidth: Volatility squeeze and expansion breakout strategy"
    category = "mean_reversion"
    tags = ["bollinger", "bandwidth", "mean_reversion"]

    def __init__(self, bb_length: int = 20, bb_mult: float = 2.0, bw_ma_length: int = 50, squeeze_factor: float = 0.75):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.bw_ma_length = bw_ma_length
        self.squeeze_factor = squeeze_factor

    def get_parameter_schema(self) -> dict:
        return {
            "bb_length": {"type": "integer", "default": 20, "minimum": 1},
            "bb_mult": {"type": "number", "default": 2.0, "minimum": 0.1},
            "bw_ma_length": {"type": "integer", "default": 50, "minimum": 1},
            "squeeze_factor": {"type": "number", "default": 0.75, "minimum": 0.01, "maximum": 1.0}
        }

    def get_warmup_periods(self) -> int:
        return max(self.bb_length, self.bw_ma_length) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()

        if len(closes) < max(self.bb_length, self.bw_ma_length):
            return None

        bb_upper, bb_lower, bb_middle = bollinger_bands(closes, self.bb_length, self.bb_mult)
        bandwidth = [(bb_upper[i] - bb_lower[i]) / bb_middle[i] * 100 if bb_middle[i] != 0 else np.nan for i in range(len(bb_upper))]
        bandwidth_ma = sma(bandwidth, self.bw_ma_length)

        current_bandwidth = bandwidth[-1]
        current_bandwidth_ma = bandwidth_ma[-1]
        current_close = closes[-1]
        current_bb_middle = bb_middle[-1]

        if len(bandwidth) < 2 or len(bandwidth_ma) < 2:
            return None

        prev_bandwidth = bandwidth[-2]
        prev_bandwidth_ma = bandwidth_ma[-2]
        prev_close = closes[-2]
        prev_bb_middle = bb_middle[-2]

        squeeze = prev_bandwidth < (prev_bandwidth_ma * self.squeeze_factor) if not np.isnan(prev_bandwidth_ma) else False
        
        if len(bandwidth) < 3 or len(bandwidth_ma) < 3:
            return None

        prev_squeeze = bandwidth[-3] < (bandwidth_ma[-3] * self.squeeze_factor) if not np.isnan(bandwidth_ma[-3]) else False

        expansion = (prev_bandwidth > prev_bandwidth_ma) and prev_squeeze if not np.isnan(prev_bandwidth_ma) else False

        long_entry = expansion and (prev_close > prev_bb_middle) if not np.isnan(prev_bb_middle) else False
        #short_entry = expansion and (prev_close < prev_bb_middle) if not np.isnan(prev_bb_middle) else False

        if long_entry:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"bandwidth": current_bandwidth, "bandwidth_ma": current_bandwidth_ma, "bb_middle": current_bb_middle, "expansion": expansion})
        #elif short_entry:
        #    return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"bandwidth": current_bandwidth, "bandwidth_ma": current_bandwidth_ma, "bb_middle": current_bb_middle, "expansion": expansion})
        else:
            return None