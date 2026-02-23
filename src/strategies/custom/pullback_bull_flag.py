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

def calculate_ema(series: pl.Series, period: int) -> pl.Series:
    alpha = 2 / (period + 1)
    ema = [series[0]]
    for i in range(1, len(series)):
        ema.append(alpha * series[i] + (1 - alpha) * ema[-1])
    return pl.Series(ema)

def calculate_rsi(series: pl.Series, period: int) -> pl.Series:
    deltas = series.diff().drop_nulls()
    seed_up = deltas[:period][deltas[:period] >= 0].sum() / period
    seed_down = -deltas[:period][deltas[:period] < 0].sum() / period
    rs = seed_up / seed_down if seed_down != 0 else 0
    rsi = [100 - (100 / (1 + rs))]

    up = deltas[period:][deltas[period:] >= 0]
    down = -deltas[period:][deltas[period:] < 0]

    for i in range(len(up)):
        upval = up[i] if not math.isnan(up[i]) else 0
        downval = down[i] if not math.isnan(down[i]) else 0

        seed_up = (seed_up * (period - 1) + upval) / period
        seed_down = (seed_down * (period - 1) + downval) / period

        rs = seed_up / seed_down if seed_down != 0 else 0
        rsi.append(100 - (100 / (1 + rs)))

    return pl.Series([None] * period + rsi)

def calculate_atr(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    tr = []
    for i in range(1, len(high)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    tr = pl.Series(tr)
    atr = [tr[:period].mean()]
    for i in range(period, len(tr)):
        atr.append((atr[-1] * (period - 1) + tr[i]) / period)
    return pl.Series([None] * period + atr)

def calculate_adx(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    up_move = high.diff().drop_nulls()
    down_move = -low.diff().drop_nulls()

    plus_dm = pl.Series([up_move[i] if up_move[i] > down_move[i] and up_move[i] > 0 else 0 for i in range(len(up_move))])
    minus_dm = pl.Series([down_move[i] if down_move[i] > up_move[i] and down_move[i] > 0 else 0 for i in range(len(down_move))])

    tr = []
    for i in range(1, len(high)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    tr = pl.Series(tr)

    plus_di = 100 * calculate_ema(plus_dm, period) / calculate_ema(tr, period)
    minus_di = 100 * calculate_ema(minus_dm, period) / calculate_ema(tr, period)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = calculate_ema(dx, period)

    return pl.Series([None] * (2 * period - 1) + adx.to_list()[period-1:])

@register
class PullbackBullFlag(BaseStrategy):
    name = "pullback_bull_flag"
    version = "1.0.0"
    description = "Pullback / Bull Flag: enter on dips in strong uptrends"
    category = "multi_factor"
    tags = ["pullback", "bull flag", "trend"]

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        rsi_period: int = 14,
        rsi_pullback_low: float = 45.0,
        rsi_pullback_high: float = 50.0,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        max_hold: int = 20,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_pullback_low = rsi_pullback_low
        self.rsi_pullback_high = rsi_pullback_high
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.max_hold = max_hold

    def get_warmup_periods(self) -> int:
        return max(self.ema_fast, self.ema_slow, self.rsi_period, self.adx_period, self.atr_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < self.get_warmup_periods():
            return None

        ema_fast_series = calculate_ema(df["close"], self.ema_fast)
        ema_slow_series = calculate_ema(df["close"], self.ema_slow)
        rsi_series = calculate_rsi(df["close"], self.rsi_period)
        adx_series = calculate_adx(df["high"], df["low"], df["close"], self.adx_period)
        atr_series = calculate_atr(df["high"], df["low"], df["close"], self.atr_period)

        # Add indicators to the DataFrame
        df = df.with_columns([
            ema_fast_series.alias("ema_fast"),
            ema_slow_series.alias("ema_slow"),
            rsi_series.alias("rsi"),
            adx_series.alias("adx"),
            atr_series.alias("atr")
        ])

        # Get the second to last row for calculations
        prev_row = df.row(len(df) - 2, named=True)

        # Ensure that the required columns exist and are not None
        if (
            prev_row["rsi"] is None or
            prev_row["adx"] is None or
            prev_row["close"] is None or
            prev_row["ema_fast"] is None or
            prev_row["ema_slow"] is None
        ):
            return None

        # Get the third to last row for calculations
        prev2_row = df.row(len(df) - 3, named=True)

        # Ensure that the required columns exist and are not None
        if (
            prev2_row["rsi"] is None
        ):
            return None

        # Strong uptrend
        strong_trend = (prev_row["adx"] > self.adx_threshold) and (prev_row["ema_fast"] > prev_row["ema_slow"])

        # Pullback: RSI dipped below threshold recently then recovered
        rsi_dipped = prev2_row["rsi"] < self.rsi_pullback_low
        rsi_recovered = prev_row["rsi"] > self.rsi_pullback_high

        # Price near fast EMA (within 2%)
        near_ema = prev_row["close"] <= prev_row["ema_fast"] * 1.02

        # Buy signal
        if strong_trend and rsi_dipped and rsi_recovered and near_ema:
            return Signal(action="buy", strength=1.0, confidence=1.0)

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "ema_fast": {
                    "type": "integer",
                    "default": 20,
                    "description": "Fast EMA period"
                },
                "ema_slow": {
                    "type": "integer",
                    "default": 50,
                    "description": "Slow EMA period"
                },
                "rsi_period": {
                    "type": "integer",
                    "default": 14,
                    "description": "RSI period"
                },
                "rsi_pullback_low": {
                    "type": "number",
                    "default": 45.0,
                    "description": "RSI must dip below this"
                },
                "rsi_pullback_high": {
                    "type": "number",
                    "default": 50.0,
                    "description": "RSI must be above this on entry"
                },
                "adx_period": {
                    "type": "integer",
                    "default": 14,
                    "description": "ADX period"
                },
                "adx_threshold": {
                    "type": "number",
                    "default": 25.0,
                    "description": "Min ADX for trending"
                },
                "atr_period": {
                    "type": "integer",
                    "default": 14,
                    "description": "ATR period"
                },
                "atr_stop_mult": {
                    "type": "number",
                    "default": 2.0,
                    "description": "ATR trailing stop multiplier"
                },
                "max_hold": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max holding days"
                },
            },
        }