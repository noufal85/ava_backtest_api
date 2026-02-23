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

def calculate_ema(series: pl.Series, period: int) -> pl.Series:
    """Calculates the Exponential Moving Average (EMA) of a series."""
    alpha = 2 / (period + 1)
    ema = [series[0]]  # Initialize EMA with the first value

    for i in range(1, len(series)):
        ema.append(alpha * series[i] + (1 - alpha) * ema[-1])

    return pl.Series(ema)

@register
class TvHammerDoji(BaseStrategy):
    name = "tv_hammer_doji"
    version = "1.0.0"
    description = "Hammer & Doji: Candlestick pattern recognition strategy"
    category = "momentum"
    tags = ["candlestick", "hammer", "doji"]

    def __init__(
        self,
        ema_period: int = 50,
        hammer_body_ratio: float = 0.5,
        hammer_wick_ratio: float = 2.0,
        doji_body_ratio: float = 0.1,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ):
        self.ema_period = ema_period
        self.hammer_body_ratio = hammer_body_ratio
        self.hammer_wick_ratio = hammer_wick_ratio
        self.doji_body_ratio = doji_body_ratio
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def get_parameter_schema(self) -> dict:
        return {
            "ema_period": {"type": "integer", "default": 50},
            "hammer_body_ratio": {"type": "number", "default": 0.5},
            "hammer_wick_ratio": {"type": "number", "default": 2.0},
            "doji_body_ratio": {"type": "number", "default": 0.1},
            "stop_loss_pct": {"type": ["number", "null"], "default": None},
            "take_profit_pct": {"type": ["number", "null"], "default": None},
        }

    def get_warmup_periods(self) -> int:
        return self.ema_period + 5  # EMA period + small buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.ema_period:
            return None

        # Calculate EMA
        closes = df["close"].to_list()
        ema_values = calculate_ema(pl.Series(closes), self.ema_period).to_list()
        ema = ema_values[-1]

        # Candlestick components
        open_price = df["open"][-1]
        high_price = df["high"][-1]
        low_price = df["low"][-1]
        close_price = df["close"][-1]

        body = abs(close_price - open_price)
        full_range = high_price - low_price
        upper_wick = high_price - max(close_price, open_price)
        lower_wick = min(close_price, open_price) - low_price

        # Hammer pattern detection
        hammer_cond1 = full_range > 0
        hammer_cond2 = lower_wick > body * self.hammer_wick_ratio
        hammer_cond3 = upper_wick < body * self.hammer_body_ratio
        hammer_cond4 = close_price < ema
        hammer = hammer_cond1 and hammer_cond2 and hammer_cond3 and hammer_cond4

        # Doji pattern detection
        doji_cond1 = full_range > 0
        doji_cond2 = body < full_range * self.doji_body_ratio
        doji = doji_cond1 and doji_cond2

        # Doji continuation signal (with momentum check)
        if len(df) >= 3:
            close_prev = df["close"][-2]
            close_prev2 = df["close"][-3]
            upward_momentum = close_prev > close_prev2
        else:
            upward_momentum = False

        doji_continue = doji and (close_price > ema) and upward_momentum

        # Generate signals
        if hammer:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"pattern": "hammer"})
        elif doji_continue:
            return Signal(action="buy", strength=0.7, confidence=0.6, metadata={"pattern": "doji_continue"})
        else:
            return None