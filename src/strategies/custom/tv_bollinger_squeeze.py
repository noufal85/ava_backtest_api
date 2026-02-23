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

def ema(series: pl.Series, period: int) -> pl.Series:
    alpha = 2 / (period + 1)
    ema = [series[0]]
    for i in range(1, len(series)):
        ema.append(alpha * series[i] + (1 - alpha) * ema[-1])
    return pl.Series(ema)

def atr(df: pl.DataFrame, period: int) -> pl.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = [max(high[0] - low[0], abs(high[0] - close[0]), abs(low[0] - close[0]))]
    for i in range(1, len(high)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    tr_series = pl.Series(tr)
    return ema(tr_series, period)

def bollinger_bands(df: pl.DataFrame, period: int, std_mult: float) -> dict:
    close = df["close"]
    middle = close.rolling(window=period).mean().fill_null(strategy="mean")
    std = close.rolling(window=period).std().fill_null(strategy="mean")
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return {"middle": middle, "upper": upper, "lower": lower}

@register
class TvBollingerSqueeze(BaseStrategy):
    name = "tv_bollinger_squeeze"
    version = "1.0.0"
    description = "Bollinger Band Squeeze: Trades breakouts from squeeze conditions"
    category = "mean_reversion"
    tags = ["bollinger bands", "squeeze", "breakout"]

    def __init__(self, bb_length: int = 20, bb_mult: float = 2.0, kc_length: int = 20, kc_mult: float = 1.5):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult

    def get_warmup_periods(self) -> int:
        return max(self.bb_length, self.kc_length) + 5

    def generate_signal(self, window) -> Signal | None:
        df = pl.concat([window.historical(), window.current_bar()])
        if len(df) < self.get_warmup_periods():
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Bollinger Bands
        bb_result = bollinger_bands(df, period=self.bb_length, std_mult=self.bb_mult)
        bb_middle = bb_result["middle"]
        bb_upper = bb_result["upper"]
        bb_lower = bb_result["lower"]

        # Keltner Channels
        kc_middle = ema(close, period=self.kc_length)
        kc_atr = atr(df, period=self.kc_length)
        kc_range = kc_atr * self.kc_mult
        kc_upper = kc_middle + kc_range
        kc_lower = kc_middle - kc_range

        # Squeeze conditions
        sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        sqz_off = ~sqz_on

        # Momentum calculation (linear regression of close - EMA over bb_length)
        close_ema_diff = close - ema(close, period=self.bb_length)

        # Linear regression slope approximation using rolling correlation
        def linreg_slope(series):
            if len(series) < self.bb_length:
                return float('nan')
            y_values = series.to_numpy()
            x_values = np.arange(len(y_values))
            x_mean = x_values.mean()
            y_mean = y_values.mean()
            numerator = np.sum((x_values - x_mean) * (y_values - y_mean))
            denominator = np.sum((x_values - x_mean) ** 2)
            if denominator == 0:
                return 0
            return numerator / denominator

        momentum = close_ema_diff.rolling(window=self.bb_length).apply(linreg_slope, eager=True)

        # Use shifted values to prevent look-ahead bias
        if len(sqz_on) < 2:
            return None

        prev_sqz_on = sqz_on[-2]
        current_sqz_off = sqz_off[-1]
        current_momentum = momentum[-1]

        # Entry conditions: squeeze release + momentum direction
        long_condition = current_sqz_off and prev_sqz_on and (current_momentum > 0)
        # short_condition = current_sqz_off and prev_sqz_on and (current_momentum < 0) # Disabled for long-only

        if long_condition:
            return Signal(
                action="buy",
                strength=1.0,
                confidence=0.8,
                metadata={
                    "bb_upper": bb_upper[-1],
                    "bb_lower": bb_lower[-1],
                    "kc_upper": kc_upper[-1],
                    "kc_lower": kc_lower[-1],
                    "momentum": current_momentum
                }
            )
        # elif short_condition:
        #     return Signal(
        #         action="sell",
        #         strength=1.0,
        #         confidence=0.8,
        #         metadata={
        #             "bb_upper": bb_upper[-1],
        #             "bb_lower": bb_lower[-1],
        #             "kc_upper": kc_upper[-1],
        #             "kc_lower": kc_lower[-1],
        #             "momentum": current_momentum
        #         }
        #     )
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "bb_length": {
                    "type": "integer",
                    "default": 20,
                    "description": "Bollinger Bands period"
                },
                "bb_mult": {
                    "type": "number",
                    "default": 2.0,
                    "description": "Bollinger Bands multiplier"
                },
                "kc_length": {
                    "type": "integer",
                    "default": 20,
                    "description": "Keltner Channels period"
                },
                "kc_mult": {
                    "type": "number",
                    "default": 1.5,
                    "description": "Keltner Channels ATR multiplier"
                }
            }
        }