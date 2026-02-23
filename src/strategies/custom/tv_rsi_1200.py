from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import numpy as np
import polars as pl

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

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return [None] * len(closes)

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    
    seed_up = sum(up for up in deltas[:period] if up > 0) / period
    seed_down = sum(abs(down) for down in deltas[:period] if down < 0) / period
    rs = seed_up / seed_down if seed_down != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    rsi_values = [None] * period + [rsi]
    
    up = seed_up
    down = seed_down
    
    for i in range(period, len(deltas)):
        delta = deltas[i]
        
        up = (up * (period - 1) + (delta if delta > 0 else 0)) / period
        down = (down * (period - 1) + (abs(delta) if delta < 0 else 0)) / period
        
        rs = up / down if down != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append(rsi)
    
    return rsi_values

def calculate_ema(closes, period=150):
    if len(closes) < period:
        return [None] * len(closes)

    ema = [None] * len(closes)
    smoothing = 2 / (period + 1)
    
    # Calculate initial SMA
    sma = sum(closes[:period]) / period
    ema[period - 1] = sma
    
    # Calculate EMA for the rest of the series
    for i in range(period, len(closes)):
        ema[i] = (closes[i] - ema[i - 1]) * smoothing + ema[i - 1]
        
    return ema

@register
class TvRsi1200(BaseStrategy):
    name = "tv_rsi_1200"
    version = "1.0.0"
    description = "RSI mean reversion with EMA trend filter and pattern recognition"
    category = "mean_reversion"
    tags = ["rsi", "ema", "mean_reversion"]

    def __init__(
        self,
        rsi_length: int = 14,
        rsi_overbought: int = 72,
        rsi_oversold: int = 28,
        ema_length: int = 150,
        stop_loss_pct: float = 10.0,
        show_long: bool = True,
        show_short: bool = False,
    ):
        self.rsi_length = rsi_length
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.ema_length = ema_length
        self.stop_loss_pct = stop_loss_pct / 100.0
        self.show_long = show_long
        self.show_short = show_short

    def get_warmup_periods(self) -> int:
        return self.ema_length + 3 # Buffer for shifts and patterns

    def generate_signal(self, window) -> Signal | None:
        historical_df = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_df, current_bar])

        closes = df["close"].to_list()
        opens = df["open"].to_list()
        
        if len(closes) < self.get_warmup_periods():
            return None

        rsi_values = calculate_rsi(closes, self.rsi_length)
        ema_values = calculate_ema(closes, self.ema_length)

        rsi = rsi_values[-1]
        ema = ema_values[-1]
        close = closes[-1]
        open_price = opens[-1]

        prev_close = closes[-2] if len(closes) > 1 else None
        prev_ema = ema_values[-2] if len(ema_values) > 1 else None
        prev_rsi = rsi_values[-2] if len(rsi_values) > 1 else None

        red_candle = close < open_price
        green_candle = close > open_price

        three_red = False
        three_green = False

        if len(closes) >= 3:
            red_candle_1 = closes[-2] < opens[-2]
            red_candle_2 = closes[-3] < opens[-3]
            green_candle_1 = closes[-2] > opens[-2]
            green_candle_2 = closes[-3] > opens[-3]
            three_red = red_candle and red_candle_1 and red_candle_2
            three_green = green_candle and green_candle_1 and green_candle_2

        if prev_close is None or prev_ema is None or prev_rsi is None:
            return None

        rsi_cross_over_oversold = (prev_rsi <= self.rsi_oversold) and (rsi > self.rsi_oversold)
        rsi_cross_under_overbought = (prev_rsi >= self.rsi_overbought) and (rsi < self.rsi_overbought)

        slack_long = prev_close > prev_ema * 1.01
        slack_short = prev_close < prev_ema * 0.99

        long_signal = (
            self.show_long and
            (prev_close > prev_ema) and
            rsi_cross_over_oversold and
            slack_long
        )

        short_signal = (
            self.show_short and
            (prev_close < prev_ema) and
            rsi_cross_under_overbought and
            slack_short
        ) #& False  # Disabled for long-only

        long_rsi_exit = rsi > self.rsi_overbought
        short_rsi_exit = rsi < self.rsi_oversold

        long_pattern_exit = (
            (close > ema) and
            (close < ema * 1.02) and
            three_red
        )
        short_pattern_exit = (
            (close < ema) and
            (close > ema * 0.99) and
            three_green
        )

        if long_signal:
            return Signal(
                action="buy",
                strength=1.0,
                confidence=1.0,
                metadata={"rsi": rsi, "ema": ema},
            )
        elif short_signal:
            return Signal(
                action="sell",
                strength=1.0,
                confidence=1.0,
                metadata={"rsi": rsi, "ema": ema},
            )
        
        # Exit Logic - Returning None implies holding
        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "rsi_length": {
                    "type": "integer",
                    "default": 14,
                    "description": "RSI period",
                },
                "rsi_overbought": {
                    "type": "integer",
                    "default": 72,
                    "description": "RSI overbought level",
                },
                "rsi_oversold": {
                    "type": "integer",
                    "default": 28,
                    "description": "RSI oversold level",
                },
                "ema_length": {
                    "type": "integer",
                    "default": 150,
                    "description": "EMA period",
                },
                "stop_loss_pct": {
                    "type": "number",
                    "default": 10.0,
                    "description": "Stop loss percentage",
                },
                "show_long": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable long trades",
                },
                "show_short": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable short trades",
                },
            },
        }