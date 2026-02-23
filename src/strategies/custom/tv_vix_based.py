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

def realized_vol(df: pl.DataFrame, period: int = 20) -> pl.Series:
    """Calculates realized volatility."""
    log_returns = np.log(df["close"]).diff().dropna()
    return log_returns.rolling(period).std()

def sma(series: pl.Series, period: int = 20) -> pl.Series:
    """Calculates Simple Moving Average."""
    return series.rolling(period).mean()

@register
class TvVixBased(BaseStrategy):
    name = "tv_vix_based"
    version = "1.0.0"
    description = "VIX-Based Strategy: Contrarian volatility-based equity strategy"
    category = "volatility"
    tags = ["volatility", "contrarian"]

    def __init__(
        self,
        vol_lookback: int = 20,
        vol_ma_length: int = 20,
        high_vol_threshold: float = 0.8,
        low_vol_threshold: float = 0.2,
        min_hold_days: int = 5,
    ):
        self.vol_lookback = vol_lookback
        self.vol_ma_length = vol_ma_length
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.min_hold_days = min_hold_days
        self.position_entry_date = None

    def get_warmup_periods(self) -> int:
        return max(self.vol_lookback, self.vol_ma_length) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < max(self.vol_lookback, self.vol_ma_length):
            return None

        closes = df["close"].to_list()
        
        # Realized Volatility
        log_returns = [math.log(closes[i] / closes[i-1]) if i > 0 else 0 for i in range(len(closes))]
        log_returns = log_returns[1:]
        
        realized_vols = []
        for i in range(self.vol_lookback, len(log_returns) + 1):
            window_data = log_returns[i-self.vol_lookback:i]
            std_dev = np.std(window_data)
            realized_vols.append(std_dev)
        
        realized_vol_series = pl.Series(realized_vols) * 100
        
        # SMA of Realized Volatility
        vol_mas = []
        for i in range(self.vol_ma_length, len(realized_vol_series) + 1):
            window_data = realized_vol_series[i-self.vol_ma_length:i]
            vol_mas.append(np.mean(window_data))
        
        if not realized_vols or not vol_mas:
            return None

        realized_vol_val = realized_vols[-1] * 100 if realized_vols else None
        vol_ma_val = vol_mas[-1] if vol_mas else None
        
        realized_vol_series = pl.Series(realized_vols) * 100
        vol_ma_series = pl.Series(vol_mas)

        # Calculate dynamic thresholds based on historical percentiles
        vol_high_level = realized_vol_series.quantile(self.high_vol_threshold)
        vol_low_level = realized_vol_series.quantile(self.low_vol_threshold)

        # Use shifted values to avoid look-ahead bias
        if len(realized_vol_series) >= 2:
            prev_vol = realized_vol_series[-1]
            prev2_vol = realized_vol_series[-2]
        else:
            return None

        # Fear condition: high vol + declining
        fear_condition = (prev_vol > vol_high_level) and (prev_vol < prev2_vol)

        # Complacency condition: low vol + rising
        complacency_condition = (prev_vol < vol_low_level) and (prev_vol > prev2_vol)

        # Entry/exit signals
        long_entry = fear_condition
        long_exit = complacency_condition

        if long_entry:
            self.position_entry_date = df["timestamp"][-1]
            return Signal(
                action="buy",
                strength=1.0,
                confidence=0.8,
                metadata={
                    "volatility_at_entry": realized_vol_val,
                    "vol_ma_at_entry": vol_ma_val,
                    "fear_level": True
                }
            )
        elif long_exit and self.position_entry_date is not None:
            days_held = (df["timestamp"][-1] - self.position_entry_date).days
            if days_held >= self.min_hold_days:
                self.position_entry_date = None
                return Signal(
                    action="sell",
                    strength=1.0,
                    confidence=0.8,
                    metadata={
                        "complacency_exit": True
                    }
                )
            else:
                return None
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "vol_lookback": {
                    "type": "integer",
                    "default": 20,
                    "description": "Volatility calculation period"
                },
                "vol_ma_length": {
                    "type": "integer",
                    "default": 20,
                    "description": "Volatility moving average period"
                },
                "high_vol_threshold": {
                    "type": "number",
                    "default": 0.8,
                    "description": "High volatility threshold - percentile"
                },
                "low_vol_threshold": {
                    "type": "number",
                    "default": 0.2,
                    "description": "Low volatility threshold - percentile"
                },
                "min_hold_days": {
                    "type": "integer",
                    "default": 5,
                    "description": "Minimum holding period"
                },
            },
        }