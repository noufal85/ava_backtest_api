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

def sma(series: list[float], period: int) -> list[float]:
    """Simple Moving Average."""
    if len(series) < period:
        return [None] * len(series)
    
    sma_values = []
    for i in range(len(series)):
        if i < period - 1:
            sma_values.append(None)
        else:
            window = series[i - period + 1:i + 1]
            sma_values.append(sum(window) / period)
    return sma_values

def bollinger_bands(series: list[float], period: int, std_dev: float) -> tuple[list[float], list[float], list[float]]:
    """Bollinger Bands."""
    basis = sma(series, period)
    upper_band = []
    lower_band = []

    for i in range(len(series)):
        if i < period - 1 or basis[i] is None:
            upper_band.append(None)
            lower_band.append(None)
        else:
            window = series[i - period + 1:i + 1]
            std = np.std(window)
            upper_band.append(basis[i] + std_dev * std)
            lower_band.append(basis[i] - std_dev * std)
    
    return upper_band, lower_band, basis

@register
class TvBbDivergence(BaseStrategy):
    name = "tv_bb_divergence"
    version = "1.0.0"
    description = "Bollinger Bands divergence strategy with band width analysis"
    category = "mean_reversion"
    tags = ["bollinger bands", "divergence", "mean reversion"]

    def __init__(self, bb_length: int = 20, bb_mult: float = 2.0, candleper: float = 30.0, tp: float = 5.0, time_stop: int = 20):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.candleper = candleper
        self.tp = tp
        self.time_stop = time_stop

    def get_parameter_schema(self) -> dict:
        return {
            "bb_length": {"type": "integer", "default": 20},
            "bb_mult": {"type": "number", "default": 2.0},
            "candleper": {"type": "number", "default": 30.0},
            "tp": {"type": "number", "default": 5.0},
            "time_stop": {"type": "integer", "default": 20},
        }

    def get_warmup_periods(self) -> int:
        return self.bb_length + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        opens = df["open"].to_list()

        # Bollinger Bands
        upper_band, lower_band, basis = bollinger_bands(closes, self.bb_length, self.bb_mult)

        if len(closes) < 2 or upper_band[-1] is None or lower_band[-1] is None or basis[-1] is None:
            return None

        # Divergence conditions
        upper_widening = upper_band[-1] > upper_band[-2] if len(upper_band) > 1 and upper_band[-2] is not None else False
        lower_narrowing = lower_band[-1] < lower_band[-2] if len(lower_band) > 1 and lower_band[-2] is not None else False
        upper_narrowing = upper_band[-1] < upper_band[-2] if len(upper_band) > 1 and upper_band[-2] is not None else False
        lower_widening = lower_band[-1] > lower_band[-2] if len(lower_band) > 1 and lower_band[-2] is not None else False

        # Price action conditions
        close_above_upper = closes[-1] > upper_band[-1]
        close_below_lower = closes[-1] < lower_band[-1]
        bullish_candle = closes[-1] > opens[-1]
        bearish_candle = closes[-1] < opens[-1]

        # Candle positioning
        candle_size = highs[-1] - lows[-1]
        buyzone = highs[-1] - (candle_size * ((100 - self.candleper) / 100))
        sellzone = lows[-1] - (candle_size * ((100 - self.candleper) / 100))

        # Entry signals
        long_signal = (
            close_above_upper and
            upper_widening and
            lower_narrowing and
            bullish_candle and
            (buyzone > upper_band[-1])
        )

        # Exit signal
        close_above_basis = closes[-1] < basis[-1]
        close_below_basis = closes[-1] > basis[-1]

        if long_signal:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"bb_upper": upper_band[-1], "bb_basis": basis[-1]})
        
        return None