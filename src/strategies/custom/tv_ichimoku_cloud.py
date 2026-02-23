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

@register
class TvIchimokuCloud(BaseStrategy):
    name = "tv_ichimoku_cloud"
    version = "1.0.0"
    description = "Ichimoku Cloud Strategy: TK cross with cloud position confirmation"
    category = "multi_factor"
    tags = ["trend", "ichimoku"]

    def __init__(self, conversion_len: int = 9, base_len: int = 26, span_b_len: int = 52, displacement: int = 26, max_hold_days: int = 60):
        self.conversion_len = conversion_len
        self.base_len = base_len
        self.span_b_len = span_b_len
        self.displacement = displacement
        self.max_hold_days = max_hold_days

    def get_parameter_schema(self) -> dict:
        return {
            "conversion_len": {"type": "integer", "default": 9, "description": "Tenkan-sen (Conversion Line) period"},
            "base_len": {"type": "integer", "default": 26, "description": "Kijun-sen (Base Line) period"},
            "span_b_len": {"type": "integer", "default": 52, "description": "Senkou Span B period"},
            "displacement": {"type": "integer", "default": 26, "description": "Cloud displacement periods"},
            "max_hold_days": {"type": "integer", "default": 60, "description": "Maximum holding period"}
        }

    def get_warmup_periods(self) -> int:
        return max(self.conversion_len, self.base_len, self.span_b_len) + self.displacement + 2 # Added 2 as buffer

    def generate_signal(self, window) -> Signal | None:
        df = pl.concat([window.historical(), window.current_bar()])
        
        conversion_len = self.conversion_len
        base_len = self.base_len
        span_b_len = self.span_b_len
        displacement = self.displacement

        high = df["high"].to_list()
        low = df["low"].to_list()
        close = df["close"].to_list()

        if len(close) < max(conversion_len, base_len, span_b_len) + displacement + 2:
            return None

        # Tenkan-sen (Conversion Line): midpoint of highest high and lowest low over conversion_len periods
        tenkan = [(max(high[i-conversion_len:i]) + min(low[i-conversion_len:i])) / 2 if i >= conversion_len else None for i in range(1, len(close)+1)]

        # Kijun-sen (Base Line): midpoint over base_len periods
        kijun = [(max(high[i-base_len:i]) + min(low[i-base_len:i])) / 2 if i >= base_len else None for i in range(1, len(close)+1)]

        # Senkou Span A (Leading Span A): average of Tenkan and Kijun
        span_a = [(tenkan[i-1] + kijun[i-1]) / 2 if tenkan[i-1] is not None and kijun[i-1] is not None else None for i in range(1, len(close)+1)]

        # Senkou Span B (Leading Span B): midpoint over span_b_len periods
        span_b = [(max(high[i-span_b_len:i]) + min(low[i-span_b_len:i])) / 2 if i >= span_b_len else None for i in range(1, len(close)+1)]

        # Cloud bounds (displaced)
        cloud_top = [max(span_a[i-1], span_b[i-1]) if span_a[i-1] is not None and span_b[i-1] is not None else None for i in range(1, len(close)+1)]
        cloud_bottom = [min(span_a[i-1], span_b[i-1]) if span_a[i-1] is not None and span_b[i-1] is not None else None for i in range(1, len(close)+1)]

        # Shift cloud values
        cloud_top_shifted = [cloud_top[i - displacement - 1] if i > displacement and cloud_top[i - displacement - 1] is not None else cloud_top[len(cloud_top)-1] if i > displacement else None for i in range(1, len(close)+1)]
        cloud_bottom_shifted = [cloud_bottom[i - displacement - 1] if i > displacement and cloud_bottom[i - displacement - 1] is not None else cloud_bottom[len(cloud_bottom)-1] if i > displacement else None for i in range(1, len(close)+1)]

        # TK Crossovers (using shifted values to prevent look-ahead)
        if len(tenkan) < 3 or len(kijun) < 3:
            return None

        prev_tenkan = tenkan[-2]
        prev2_tenkan = tenkan[-3]
        prev_kijun = kijun[-2]
        prev2_kijun = kijun[-3]

        # TK Cross Up: Tenkan crosses above Kijun
        tk_cross_up = (prev_tenkan > prev_kijun) and (prev2_tenkan <= prev2_kijun)
        
        # TK Cross Down: Tenkan crosses below Kijun
        tk_cross_down = (prev_tenkan < prev_kijun) and (prev2_tenkan >= prev2_kijun)

        # Cloud position (use shifted values)
        prev_close = close[-2]
        prev_cloud_top = cloud_top_shifted[-2]
        prev_cloud_bottom = cloud_bottom_shifted[-2]

        # Price above/below cloud
        above_cloud = prev_close > prev_cloud_top if prev_cloud_top is not None else False
        below_cloud = prev_close < prev_cloud_bottom if prev_cloud_bottom is not None else False

        # Entry signals
        # Long: TK cross up + price above cloud
        long_signal = tk_cross_up and above_cloud
        
        # Short: TK cross down + price below cloud (disabled for long-only mode)
        short_signal = False  # Long-only mode as specified

        # Exit signals - opposite TK cross
        long_exit = tk_cross_down
        short_exit = tk_cross_up

        if long_signal:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"tenkan": tenkan[-1], "kijun": kijun[-1], "cloud_top": cloud_top_shifted[-1], "cloud_bottom": cloud_bottom_shifted[-1]})
        elif short_signal:
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"tenkan": tenkan[-1], "kijun": kijun[-1], "cloud_top": cloud_top_shifted[-1], "cloud_bottom": cloud_bottom_shifted[-1]})
        else:
            return None