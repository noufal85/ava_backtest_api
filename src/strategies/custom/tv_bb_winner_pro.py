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

def sma(series: pl.Series, period: int) -> pl.Series:
    """Simple Moving Average."""
    return series.rolling_mean(window_size=period, min_periods=period)

def ema(series: pl.Series, period: int) -> pl.Series:
    """Exponential Moving Average."""
    alpha = 2 / (period + 1)
    ema = [series[0]]
    for i in range(1, len(series)):
        ema.append(alpha * series[i] + (1 - alpha) * ema[-1])
    return pl.Series(ema)

def rsi(series: pl.Series, period: int) -> pl.Series:
    """Relative Strength Index."""
    delta = series.diff().fill_null(0.0)
    up, down = delta.clone(), delta.clone()
    up = up.clip(lower_bound=0)
    down = down.clip(upper_bound=0).abs()

    avg_gain = ema(up, period)
    avg_loss = ema(down, period)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def bollinger_bands(series: pl.Series, period: int, std_dev: float) -> tuple[pl.Series, pl.Series, pl.Series]:
    """Bollinger Bands."""
    middle_band = series.rolling_mean(window_size=period, min_periods=period)
    std = series.rolling_std(window_size=period, min_periods=period)
    upper_band = middle_band + std_dev * std
    lower_band = middle_band - std_dev * std
    return upper_band, middle_band, lower_band

def atr(df: pl.DataFrame, period: int) -> pl.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).fill_null(0.0)
    tr3 = (close.shift(1) - low).fill_null(0.0)
    true_range = pl.max([tr1, tr2, tr3])
    return true_range.rolling_mean(window_size=period, min_periods=period)

@register
class TvBbWinnerPro(BaseStrategy):
    name = "tv_bb_winner_pro"
    version = "1.0.0"
    description = "BB Winner PRO: Advanced Bollinger Bands mean reversion with multiple filters"
    category = "mean_reversion"
    tags = ["mean_reversion", "bollinger_bands", "rsi", "aroon", "moving_average"]

    def __init__(
        self,
        bb_length: int = 20,
        bb_std: float = 2.0,
        candle_pct: float = 0.30,
        include_shadow: bool = False,
        use_rsi: bool = True,
        rsi_long_threshold: int = 45,
        rsi_short_threshold: int = 55,
        rsi_length: int = 14,
        use_aroon: bool = False,
        aroon_length: int = 288,
        aroon_confirmation: int = 90,
        aroon_stop: int = 70,
        use_ma: bool = True,
        ma_type: str = "EMA",
        ma_length: int = 200,
        use_sl: bool = True,
        sl_type: str = "Percent",
        sl_percent: float = 0.07,
        sl_atr_mult: float = 7.0,
        sl_atr_length: int = 14,
        close_early: bool = True,
    ):
        self.bb_length = bb_length
        self.bb_std = bb_std
        self.candle_pct = candle_pct
        self.include_shadow = include_shadow
        self.use_rsi = use_rsi
        self.rsi_long_threshold = rsi_long_threshold
        self.rsi_short_threshold = rsi_short_threshold
        self.rsi_length = rsi_length
        self.use_aroon = use_aroon
        self.aroon_length = aroon_length
        self.aroon_confirmation = aroon_confirmation
        self.aroon_stop = aroon_stop
        self.use_ma = use_ma
        self.ma_type = ma_type
        self.ma_length = ma_length
        self.use_sl = use_sl
        self.sl_type = sl_type
        self.sl_percent = sl_percent
        self.sl_atr_mult = sl_atr_mult
        self.sl_atr_length = sl_atr_length
        self.close_early = close_early

    def get_warmup_periods(self) -> int:
        return max(self.bb_length, self.rsi_length, self.ma_length, self.aroon_length, self.sl_atr_length) + 5

    def generate_signal(self, window) -> Signal | None:
        df = pl.concat([window.historical(), window.current_bar()])
        if len(df) < self.get_warmup_periods():
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = bollinger_bands(close, self.bb_length, self.bb_std)
        rsi_val = rsi(close, self.rsi_length)

        # Moving Average
        if self.ma_type == "EMA":
            ma = ema(close, self.ma_length)
        else:
            ma = sma(close, self.ma_length)

        # Aroon Up
        if self.use_aroon:
            highest_bars = high.rolling_apply(lambda x: self.aroon_length - np.argmax(x), window_size=self.aroon_length + 1, min_periods=self.aroon_length + 1).fill_null(0.0)
            aroon_up = 100 * (self.aroon_length - highest_bars) / self.aroon_length
        else:
            aroon_up = pl.Series([0.0] * len(df))

        atr_val = atr(df, self.sl_atr_length)

        # Candle body calculations
        body_size = (close - open_).abs()
        candle_range = high - low

        # Calculate penetration zones
        if self.include_shadow:
            candle_size = candle_range
        else:
            candle_size = body_size

        buy_zone = low + candle_size * self.candle_pct
        sell_zone = high - candle_size * self.candle_pct

        # Use shifted values to prevent look-ahead bias
        prev_bb_upper = bb_upper.shift(1).fill_null(0.0)
        prev_bb_lower = bb_lower.shift(1).fill_null(0.0)
        prev_rsi = rsi_val.shift(1).fill_null(0.0)
        prev_ma = ma.shift(1).fill_null(0.0)
        prev_close = close.shift(1).fill_null(0.0)
        prev_open = open_.shift(1).fill_null(0.0)
        prev_aroon_up = aroon_up.shift(1).fill_null(0.0)
        prev_buy_zone = buy_zone.shift(1).fill_null(0.0)
        prev_sell_zone = sell_zone.shift(1).fill_null(0.0)

        # Signal conditions
        # Long: buy_zone penetrates lower BB, candle is red, filters pass
        long_bb_condition = buy_zone.last() < prev_bb_lower.last()
        long_candle_condition = close.last() < open_.last()  # Red candle

        # RSI filter for long
        long_rsi_condition = not self.use_rsi or (prev_rsi.last() < self.rsi_long_threshold)

        # Aroon filter for long
        long_aroon_condition = not self.use_aroon or (prev_aroon_up.last() > self.aroon_confirmation)

        # MA filter for long
        long_ma_condition = not self.use_ma or (prev_close.last() > prev_ma.last())

        long_signal = (
            long_bb_condition and
            long_candle_condition and
            long_rsi_condition and
            long_aroon_condition and
            long_ma_condition
        )

        # Short: sell_zone penetrates upper BB, candle is green, filters pass
        short_bb_condition = sell_zone.last() > prev_bb_upper.last()
        short_candle_condition = close.last() > open_.last()  # Green candle

        # RSI filter for short
        short_rsi_condition = not self.use_rsi or (prev_rsi.last() > self.rsi_short_threshold)

        # Aroon filter for short
        short_aroon_condition = not self.use_aroon or (prev_aroon_up.last() > self.aroon_confirmation)

        # MA filter for short
        short_ma_condition = not self.use_ma or (prev_close.last() < prev_ma.last())

        short_signal = (
            short_bb_condition and
            short_candle_condition and
            short_rsi_condition and
            short_aroon_condition and
            short_ma_condition
        )

        # Exit conditions (when price touches opposite BB)
        long_exit = buy_zone.last() > prev_bb_upper.last()
        short_exit = sell_zone.last() < prev_bb_lower.last()

        # Early exit when touching opposite BB and in profit
        long_early_exit = close.last() > prev_bb_upper.last()
        short_early_exit = close.last() < prev_bb_lower.last()

        if long_signal:
            return Signal(action="buy", strength=1.0, confidence=1.0)
        elif short_signal:
            return Signal(action="sell", strength=1.0, confidence=1.0)
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "bb_length": {"type": "integer", "default": 20},
                "bb_std": {"type": "number", "default": 2.0},
                "candle_pct": {"type": "number", "default": 0.30},
                "include_shadow": {"type": "boolean", "default": False},
                "use_rsi": {"type": "boolean", "default": True},
                "rsi_long_threshold": {"type": "integer", "default": 45},
                "rsi_short_threshold": {"type": "integer", "default": 55},
                "rsi_length": {"type": "integer", "default": 14},
                "use_aroon": {"type": "boolean", "default": False},
                "aroon_length": {"type": "integer", "default": 288},
                "aroon_confirmation": {"type": "integer", "default": 90},
                "aroon_stop": {"type": "integer", "default": 70},
                "use_ma": {"type": "boolean", "default": True},
                "ma_type": {"type": "string", "default": "EMA", "enum": ["EMA", "SMA"]},
                "ma_length": {"type": "integer", "default": 200},
                "use_sl": {"type": "boolean", "default": True},
                "sl_type": {"type": "string", "default": "Percent", "enum": ["Percent", "ATR"]},
                "sl_percent": {"type": "number", "default": 0.07},
                "sl_atr_mult": {"type": "number", "default": 7.0},
                "sl_atr_length": {"type": "integer", "default": 14},
                "close_early": {"type": "boolean", "default": True},
            },
        }