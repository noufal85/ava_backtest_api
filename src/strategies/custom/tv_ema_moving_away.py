"""tv_ema_moving_away — converted from trading-backtester."""
import polars as pl
import numpy as np
import math
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register

@register
class TvEmaMovingAway(BaseStrategy):
    name = "tv_ema_moving_away"
    name: str = "tv_ema_moving_away"
    version: str = "1.0.0"
    description: str = "EMA Moving Away Strategy: Contrarian entries when price moves away from EMA"
    category: str = "trend"
    tags: list[str] = []

    def __init__(self, length: int = 55, moving_away_pct: float = 2.0, stop_loss_pct: float = 2.0, max_candle_body_pct: float = 2.0):
        self.length = length
        self.moving_away_pct = moving_away_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_candle_body_pct = max_candle_body_pct

    def get_warmup_periods(self) -> int:
        return self.length + 5  # EMA period + buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.length + 1:
            return None

        closes = df["close"]
        opens = df["open"]

        # EMA
        ema_values = ema(closes, self.length)
        df = df.with_columns(pl.Series(name="ema", values=ema_values))

        # Candle body percentage
        candle_body_pct = (abs(closes - opens) / opens * 100)
        df = df.with_columns(pl.Series(name="candle_body_pct", values=candle_body_pct))

        # Check if candle is green
        is_green = closes > opens
        df = df.with_columns(pl.Series(name="is_green", values=is_green))

        # Last 4 candles green (rolling check)
        last4_green = [all(df["is_green"][max(0, i-3):i+1]) for i in range(len(df))]
        df = df.with_columns(pl.Series(name="last4_green", values=last4_green))

        # Use shifted values to avoid look-ahead bias
        prev_ema = df["ema"][-2]
        prev_candle_body_pct = df["candle_body_pct"][-2]
        prev_last4_green = df["last4_green"][-2]
        current_close = df["close"][-1]

        # Entry conditions
        long_entry = (
            (current_close <= prev_ema * (1 - self.moving_away_pct/100)) and
            (prev_candle_body_pct < self.max_candle_body_pct)
        )

        short_entry = (
            (current_close >= prev_ema * (1 + self.moving_away_pct/100)) and
            (prev_candle_body_pct < self.max_candle_body_pct) and
            (not prev_last4_green)
        )

        # Exit conditions - price returns to EMA
        long_exit = current_close >= df["ema"][-1]
        short_exit = current_close <= df["ema"][-1]

        if long_entry:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"ema": df["ema"][-1]})
        elif short_entry:
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"ema": df["ema"][-1]})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "length": {
                    "type": "integer",
                    "default": 55,
                    "description": "EMA period"
                },
                "moving_away_pct": {
                    "type": "number",
                    "default": 2.0,
                    "description": "Required percentage away from EMA"
                },
                "stop_loss_pct": {
                    "type": "number",
                    "default": 2.0,
                    "description": "Stop loss percentage"
                },
                "max_candle_body_pct": {
                    "type": "number",
                    "default": 2.0,
                    "description": "Maximum candle body percentage"
                }
            },
            "required": ["length", "moving_away_pct", "stop_loss_pct", "max_candle_body_pct"]
        }