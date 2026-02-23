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

def rsi(closes: list[float], period: int = 14) -> list[float]:
    """Calculates the Relative Strength Index (RSI)."""
    if len(closes) < period + 1:
        return [np.nan] * len(closes)

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    avg_gain = np.mean([d for d in deltas[:period] if d > 0])
    avg_loss = np.mean([-d for d in deltas[:period] if d < 0])

    rsi_values = [np.nan] * period
    if avg_loss == 0:
        rsi_values.append(100.0 if avg_gain > 0 else 0.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs)))

    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
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

def bollinger_bands(closes: list[float], period: int = 20, std_dev: float = 2.0) -> tuple[list[float], list[float], list[float]]:
    """Calculates Bollinger Bands."""
    if len(closes) < period:
        return [np.nan] * len(closes), [np.nan] * len(closes), [np.nan] * len(closes)

    sma_values = []
    upper_band_values = []
    lower_band_values = []

    for i in range(len(closes)):
        if i < period - 1:
            sma_values.append(np.nan)
            upper_band_values.append(np.nan)
            lower_band_values.append(np.nan)
        else:
            window = closes[i - period + 1:i + 1]
            sma = np.mean(window)
            std = np.std(window)
            upper_band = sma + std_dev * std
            lower_band = sma - std_dev * std

            sma_values.append(sma)
            upper_band_values.append(upper_band)
            lower_band_values.append(lower_band)

    return upper_band_values, lower_band_values, sma_values

@register
class TvMeanReversionRsiBb(BaseStrategy):
    name = "tv_mean_reversion_rsi_bb"
    version = "1.0.0"
    description = "Mean Reversion RSI + BB: Long-only mean reversion strategy"
    category = "mean_reversion"
    tags = ["mean_reversion", "rsi", "bollinger_bands"]

    def __init__(
        self,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        rsi_neutral_low: int = 45,
        rsi_neutral_high: int = 55,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ):
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_neutral_low = rsi_neutral_low
        self.rsi_neutral_high = rsi_neutral_high
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def get_warmup_periods(self) -> int:
        return max(self.rsi_period, self.bb_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        volumes = df["volume"].to_list()

        if len(closes) < max(self.rsi_period, self.bb_period):
            return None

        rsi_values = rsi(closes, self.rsi_period)
        upper_band, lower_band, middle_band = bollinger_bands(closes, self.bb_period, self.bb_std)

        current_rsi = rsi_values[-1]
        current_close = closes[-1]
        current_lower_band = lower_band[-1]

        if (
            current_rsi < self.rsi_oversold and current_close < current_lower_band
        ):
            return Signal(
                action="buy",
                strength=1.0,
                confidence=1.0,
                metadata={
                    "rsi": current_rsi,
                    "bb_lower": current_lower_band,
                },
            )
        elif (
            current_rsi > self.rsi_neutral_low and current_rsi < self.rsi_neutral_high
        ):
            return Signal(
                action="sell",
                strength=1.0,
                confidence=1.0,
                metadata={
                    "rsi": current_rsi,
                },
            )
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "rsi_period": {
                    "type": "integer",
                    "default": 14,
                    "description": "RSI calculation period",
                },
                "bb_period": {
                    "type": "integer",
                    "default": 20,
                    "description": "Bollinger Bands period",
                },
                "bb_std": {
                    "type": "number",
                    "default": 2.0,
                    "description": "Bollinger Bands standard deviation multiplier",
                },
                "rsi_oversold": {
                    "type": "integer",
                    "default": 30,
                    "description": "RSI oversold threshold",
                },
                "rsi_overbought": {
                    "type": "integer",
                    "default": 70,
                    "description": "RSI overbought threshold",
                },
                "rsi_neutral_low": {
                    "type": "integer",
                    "default": 45,
                    "description": "RSI neutral zone lower bound",
                },
                "rsi_neutral_high": {
                    "type": "integer",
                    "default": 55,
                    "description": "RSI neutral zone upper bound",
                },
                "stop_loss_pct": {
                    "type": ["number", "null"],
                    "default": None,
                    "description": "Optional stop loss percentage",
                },
                "take_profit_pct": {
                    "type": ["number", "null"],
                    "default": None,
                    "description": "Optional take profit percentage",
                },
            },
        }