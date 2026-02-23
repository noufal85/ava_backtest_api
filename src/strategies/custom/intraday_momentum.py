"""intraday_momentum — converted from trading-backtester."""
import polars as pl
import numpy as np
import math
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register

@register
class IntradayMomentum(BaseStrategy):
    name = "intraday_momentum"
    name: str = "intraday_momentum"
    version: str = "1.0.0"
    description: str = "Intraday Momentum: 15-min momentum with daily ATR profit targets"
    category: str = "momentum"
    tags: list[str] = []

    def __init__(
        self,
        move_threshold_pct: float = 0.01,
        atr_period: int = 14,
        tp1_atr_mult: float = 1.0,
        tp2_atr_mult: float = 2.0,
    ):
        self.move_threshold_pct = move_threshold_pct
        self.atr_period = atr_period
        self.tp1_atr_mult = tp1_atr_mult
        self.tp2_atr_mult = tp2_atr_mult

    def get_parameter_schema(self) -> dict:
        return {
            "move_threshold_pct": {"type": "number", "default": 0.01},
            "atr_period": {"type": "integer", "default": 14},
            "tp1_atr_mult": {"type": "number", "default": 1.0},
            "tp2_atr_mult": {"type": "number", "default": 2.0},
        }

    def get_warmup_periods(self) -> int:
        return self.atr_period + 5 # ATR period + small buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if df.height < self.atr_period + 1:
            return None

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        opens = df["open"].to_list()
        timestamps = df["timestamp"].to_list()

        # Calculate daily open
        daily_opens = {}
        for i in range(len(timestamps)):
            date = timestamps[i].date()
            if date not in daily_opens:
                daily_opens[date] = opens[i]

        daily_open_prices = []
        for timestamp in timestamps:
            date = timestamp.date()
            daily_open_prices.append(daily_opens[date])

        # Calculate percent move from daily open
        pct_from_open = [(close - daily_open) / daily_open if daily_open != 0 else 0 for close, daily_open in zip(closes, daily_open_prices)]

        # Calculate ATR
        atr_values = atr(highs, lows, closes, self.atr_period)

        # Get previous bar data
        if len(pct_from_open) < 2 or len(atr_values) < 2:
            return None

        prev_pct_move = pct_from_open[-2]
        prev_atr = atr_values[-2]

        # Generate signals
        if prev_pct_move > self.move_threshold_pct:
            return Signal(
                action="buy",
                strength=abs(prev_pct_move),
                confidence=1.0,
                metadata={"daily_atr": prev_atr, "pct_move": prev_pct_move},
            )
        elif prev_pct_move < -self.move_threshold_pct:
            return Signal(
                action="sell",
                strength=abs(prev_pct_move),
                confidence=1.0,
                metadata={"daily_atr": prev_atr, "pct_move": prev_pct_move},
            )
        else:
            return None