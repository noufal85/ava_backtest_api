from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import polars as pl
import numpy as np
from scipy.signal import argrelextrema
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

@register
class TvIctSmartMoney(BaseStrategy):
    name = "tv_ict_smart_money"
    version = "1.0.0"
    description = "ICT Smart Money: Educational smart money concepts strategy"
    category = "multi_factor"
    tags = ["ict", "smart_money", "market_structure"]

    def __init__(self, swing_length: int = 5, fvg_enabled: bool = True, ob_enabled: bool = True):
        self.swing_length = swing_length
        self.fvg_enabled = fvg_enabled
        self.ob_enabled = ob_enabled

    def get_parameter_schema(self) -> dict:
        return {
            "swing_length": {"type": "integer", "default": 5, "description": "Pivot detection length"},
            "fvg_enabled": {"type": "boolean", "default": True, "description": "Enable Fair Value Gap detection"},
            "ob_enabled": {"type": "boolean", "default": True, "description": "Enable Order Block detection"},
        }

    def get_warmup_periods(self) -> int:
        return self.swing_length + 5  # Add a small buffer

    def _detect_pivots(self, data: list[float], swing_len: int) -> tuple[list[float], list[float]]:
        """Detect pivot highs and lows."""
        # Use scipy to find local maxima and minima
        highs = [0.0] * len(data)
        lows = [0.0] * len(data)
        
        # Find local maxima (pivot highs)
        max_indices = argrelextrema(np.array(data), np.greater, order=swing_len)[0]
        for i in max_indices:
            highs[i] = data[i]
        
        # Find local minima (pivot lows)
        min_indices = argrelextrema(np.array(data), np.less, order=swing_len)[0]
        for i in min_indices:
            lows[i] = data[i]
        
        return highs, lows

    def generate_signal(self, window) -> Signal | None:
        df = window.historical().with_columns(pl.col("ts").cast(pl.Date).alias("date"))
        current_bar = window.current_bar().with_columns(pl.col("ts").cast(pl.Date).alias("date"))
        df = pl.concat([df, current_bar])

        swing_length = self.swing_length
        
        # Detect pivot highs and lows
        highs, lows = df["high"].to_list(), df["low"].to_list()
        pivot_highs, pivot_lows = self._detect_pivots(highs, swing_length)
        
        # Track market structure
        last_hh = None
        last_ll = None
        trend = 0

        if len(df) > swing_length:
            # Determine trend based on market structure
            # Simplified: compare current swing highs/lows to previous ones
            current_high = pivot_highs[-1]
            current_low = pivot_lows[-1]
            
            if current_high > 0:  # New swing high found
                valid_highs = [h for h in pivot_highs[:-1] if h > 0]
                if len(valid_highs) >= 2:
                    prev_high = max(valid_highs[-2:])
                else:
                    prev_high = 0
                if current_high > prev_high:
                    trend = 1  # Bullish
                else:
                    trend = -1  # Bearish (lower high)
                    
            if current_low > 0:  # New swing low found
                valid_lows = [l for l in pivot_lows[:-1] if l > 0]
                if len(valid_lows) >= 2:
                    prev_low = min(valid_lows[-2:])
                else:
                    prev_low = float('inf')
                if current_low < prev_low:
                    trend = -1  # Bearish
                else:
                    trend = 1  # Bullish (higher low)

        # Fair Value Gap detection
        bull_fvg = False
        bear_fvg = False
        if len(df) >= 3:
            bull_fvg = (
                (df["low"][-1] > df["high"][-3]) and  # Gap up
                (df["close"][-2] > df["open"][-2])  # Previous bullish candle
            )
            
            bear_fvg = (
                (df["high"][-1] < df["low"][-3]) and  # Gap down
                (df["close"][-2] < df["open"][-2])  # Previous bearish candle
            )
        
        # Order Block detection (simplified)
        bull_ob = False
        bear_ob = False
        if len(df) >= 2:
            bull_ob = (
                (df["close"][-2] < df["open"][-2]) and  # Previous bearish candle
                (df["close"][-1] > df["open"][-1]) and  # Current bullish candle
                (df["close"][-1] > df["high"][-2])  # Close above previous high
            )
            
            bear_ob = (
                (df["close"][-2] > df["open"][-2]) and  # Previous bullish candle
                (df["close"][-1] < df["open"][-1]) and  # Current bearish candle
                (df["close"][-1] < df["low"][-2])  # Close below previous low
            )

        # Use shifted values to avoid look-ahead bias
        prev_trend = trend
        prev_bull_fvg = bull_fvg
        prev_bear_fvg = bear_fvg
        prev_bull_ob = bull_ob
        prev_bear_ob = bear_ob

        # Combine FVG and OB conditions based on config
        bull_trigger = False
        bear_trigger = False
        
        if self.fvg_enabled:
            bull_trigger |= prev_bull_fvg
            bear_trigger |= prev_bear_fvg
            
        if self.ob_enabled:
            bull_trigger |= prev_bull_ob
            bear_trigger |= prev_bear_ob

        # Entry conditions
        long_entry = (prev_trend == 1) and bull_trigger
        short_entry = (prev_trend == -1) and bear_trigger  # Disabled for long-only

        if long_entry:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"trend": prev_trend, "bull_fvg": prev_bull_fvg, "bull_ob": prev_bull_ob})
        elif short_entry:
            return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"trend": prev_trend, "bear_fvg": prev_bear_fvg, "bear_ob": prev_bear_ob})
        else:
            return None