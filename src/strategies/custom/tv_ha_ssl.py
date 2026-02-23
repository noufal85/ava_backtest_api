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

def sma(close: list[float], period: int) -> list[float]:
    """Simple Moving Average."""
    if len(close) < period:
        return [None] * len(close)
    
    sma_values = []
    for i in range(len(close)):
        if i < period - 1:
            sma_values.append(None)
        else:
            window = close[i - period + 1:i + 1]
            sma_values.append(sum(window) / period)
    return sma_values

@register
class TvHaSsl(BaseStrategy):
    name = "tv_ha_ssl"
    version = "1.0.0"
    description = "Heikin Ashi SSL: Long-only trend following strategy"
    category = "multi_factor"
    tags = ["trend following", "heikin ashi", "ssl"]

    def __init__(
        self,
        ssl_period: int = 3,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float = 0.003,
        exit_on_ssl_cross: bool = False,
    ):
        self.ssl_period = ssl_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.exit_on_ssl_cross = exit_on_ssl_cross

    def get_parameter_schema(self) -> dict:
        return {
            "ssl_period": {"type": "integer", "default": 3, "minimum": 1},
            "stop_loss_pct": {"type": "number", "default": 0.01, "minimum": 0.0},
            "take_profit_pct": {"type": "number", "default": 0.003, "minimum": 0.0},
            "exit_on_ssl_cross": {"type": "boolean", "default": False},
        }

    def get_warmup_periods(self) -> int:
        return self.ssl_period + 5  # Add a small buffer

    def generate_signal(self, window) -> Signal | None:
        df = window.historical().to_pandas()
        current_bar = window.current_bar().to_pandas().iloc[0]
        df = pl.concat([window.historical(), window.current_bar()]).to_pandas()

        ssl_period = self.ssl_period

        # Calculate Heikin Ashi values
        ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        ha_open = np.zeros(len(df))
        ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
        
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        
        ha_high = np.maximum.reduce([df["high"], ha_open, ha_close])
        ha_low = np.minimum.reduce([df["low"], ha_open, ha_close])

        # Calculate SSL levels
        sma_high = sma(ha_high.tolist(), period=ssl_period)
        sma_low = sma(ha_low.tolist(), period=ssl_period)

        # Calculate Hlv (High-Low value indicator)
        hlv = np.zeros(len(df), dtype=int)
        hlv[0] = 0
        
        for i in range(1, len(df)):
            if ha_close[i] > sma_high[i]:
                hlv[i] = 1
            elif ha_close[i] < sma_low[i]:
                hlv[i] = -1
            else:
                hlv[i] = hlv[i-1]

        # Calculate SSL Up and Down
        ssl_down = np.where(np.array(hlv) < 0, sma_high, sma_low)
        ssl_up = np.where(np.array(hlv) < 0, sma_low, sma_high)

        # Use shifted values to avoid look-ahead bias
        if len(ssl_up) < 3:
            return None

        prev_ssl_up = ssl_up[-2]
        prev_ssl_down = ssl_down[-2]
        prev_ssl_up_2 = ssl_up[-3]
        prev_ssl_down_2 = ssl_down[-3]

        # Long signal: SSL Up crosses above SSL Down
        long_signal = (
            (prev_ssl_up_2 <= prev_ssl_down_2) and 
            (prev_ssl_up > prev_ssl_down)
        )

        if long_signal:
            return Signal(
                action="buy",
                strength=1.0,
                confidence=1.0,
                metadata={
                    "ssl_up": ssl_up[-1],
                    "ssl_down": ssl_down[-1],
                    "ha_close": ha_close[-1]
                }
            )

        return None