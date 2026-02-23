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

def rsi(closes: list[float], period: int) -> list[float]:
    """Calculates the Relative Strength Index (RSI)."""
    if len(closes) < period + 1:
        return [np.nan] * len(closes)

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    seed_up = sum(d for d in deltas[:period] if d > 0) / period
    seed_down = -sum(d for d in deltas[:period] if d < 0) / period
    rs = seed_up / seed_down if seed_down != 0 else 0
    rsi_first = 100 - 100 / (1 + rs)

    rsi_values = [np.nan] * period + [rsi_first]
    up = seed_up
    down = seed_down

    for i in range(period, len(deltas)):
        delta = deltas[i]
        up = (up * (period - 1) + (delta if delta > 0 else 0)) / period
        down = (down * (period - 1) - (delta if delta < 0 else 0)) / period
        rs = up / down if down != 0 else 0
        rsi_val = 100 - 100 / (1 + rs)
        rsi_values.append(rsi_val)

    return rsi_values

def bollinger_bands(closes: list[float], period: int, std_dev: float) -> tuple[list[float], list[float], list[float]]:
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
            sma = sum(window) / period
            sma_values.append(sma)
            std = math.sqrt(sum([(x - sma) ** 2 for x in window]) / period)
            upper_band = sma + std_dev * std
            lower_band = sma - std_dev * std
            upper_band_values.append(upper_band)
            lower_band_values.append(lower_band)

    return upper_band_values, lower_band_values, sma_values

@register
class TvFlawlessVictory(BaseStrategy):
    name = "tv_flawless_victory"
    version = "1.0.0"
    description = "Flawless Victory: RSI and Bollinger Bands mean reversion strategy"
    category = "multi_factor"
    tags = ["mean_reversion", "rsi", "bollinger_bands"]

    def __init__(
        self,
        version: int = 2,
        rsi_period: int = 14,
        bb_period: int = 17,
        bb_std: float = 1.0,
        rsi_lower: int = 42,
        rsi_upper: int = 76,
        mfi_period: int = 14,
        mfi_lower: int = 60,
        mfi_upper: int = 64,
        stop_loss_pct: float = 0.066,
        take_profit_pct: float = 0.023,
    ):
        self.version = version
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.mfi_period = mfi_period
        self.mfi_lower = mfi_lower
        self.mfi_upper = mfi_upper
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def get_parameter_schema(self) -> dict:
        return {
            "version": {"type": "integer", "default": 2},
            "rsi_period": {"type": "integer", "default": 14},
            "bb_period": {"type": "integer", "default": 17},
            "bb_std": {"type": "number", "default": 1.0},
            "rsi_lower": {"type": "integer", "default": 42},
            "rsi_upper": {"type": "integer", "default": 76},
            "mfi_period": {"type": "integer", "default": 14},
            "mfi_lower": {"type": "integer", "default": 60},
            "mfi_upper": {"type": "integer", "default": 64},
            "stop_loss_pct": {"type": "number", "default": 0.066},
            "take_profit_pct": {"type": "number", "default": 0.023},
        }

    def get_warmup_periods(self) -> int:
        return max(self.rsi_period, self.bb_period, self.mfi_period) + 5

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        volumes = df["volume"].to_list()

        if len(closes) < max(self.rsi_period, self.bb_period, self.mfi_period):
            return None

        rsi_values = rsi(closes, self.rsi_period)
        bb_upper, bb_lower, _ = bollinger_bands(closes, self.bb_period, self.bb_std)

        current_close = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else None
        current_rsi = rsi_values[-1]
        prev_rsi = rsi_values[-2] if len(rsi_values) > 1 else None
        prev_bb_upper = bb_upper[-2] if len(bb_upper) > 1 else None
        prev_bb_lower = bb_lower[-2] if len(bb_lower) > 1 else None

        if prev_close is None or prev_rsi is None or prev_bb_upper is None or prev_bb_lower is None:
            return None

        if self.version == 1:
            buy_trigger = prev_close < prev_bb_lower
            buy_guard = prev_rsi < self.rsi_lower
            sell_trigger = prev_close > prev_bb_upper
            sell_guard = prev_rsi > self.rsi_upper

            if buy_trigger and buy_guard:
                return Signal(action="buy", strength=1.0, confidence=0.8)
            elif sell_trigger and sell_guard:
                return Signal(action="sell", strength=1.0, confidence=0.8)
            else:
                return None

        elif self.version == 2:
            buy_trigger = prev_close < prev_bb_lower
            buy_guard = prev_rsi < self.rsi_lower
            sell_trigger = prev_close > prev_bb_upper
            sell_guard = prev_rsi > self.rsi_upper

            if buy_trigger and buy_guard:
                return Signal(action="buy", strength=1.0, confidence=0.8)
            elif sell_trigger and sell_guard:
                return Signal(action="sell", strength=1.0, confidence=0.8)
            else:
                return None

        elif self.version == 3:
            typical_prices = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
            raw_money_flows = [typical_prices[i] * volumes[i] for i in range(len(closes))]

            positive_flows = [0.0] * len(closes)
            negative_flows = [0.0] * len(closes)

            for i in range(1, len(closes)):
                if typical_prices[i] > typical_prices[i - 1]:
                    positive_flows[i] = raw_money_flows[i]
                elif typical_prices[i] < typical_prices[i - 1]:
                    negative_flows[i] = raw_money_flows[i]

            positive_mf = [sum(positive_flows[max(0, i - self.mfi_period + 1):i + 1]) if i >= self.mfi_period - 1 else np.nan for i in range(len(positive_flows))]
            negative_mf = [sum(negative_flows[max(0, i - self.mfi_period + 1):i + 1]) if i >= self.mfi_period - 1 else np.nan for i in range(len(negative_flows))]

            money_flow_ratios = [positive_mf[i] / negative_mf[i] if negative_mf[i] != 0 and not np.isnan(positive_mf[i]) and not np.isnan(negative_mf[i]) else np.nan for i in range(len(positive_mf))]
            mfi_values = [100 - (100 / (1 + money_flow_ratios[i])) if not np.isnan(money_flow_ratios[i]) else 50 for i in range(len(money_flow_ratios))]

            current_mfi = mfi_values[-1]
            prev_mfi = mfi_values[-2] if len(mfi_values) > 1 else None

            if prev_mfi is None:
                return None

            buy_trigger = prev_close < prev_bb_lower
            buy_guard = prev_mfi < self.mfi_lower
            sell_trigger = prev_close > prev_bb_upper
            sell_guard = (prev_rsi > self.rsi_upper) and (prev_mfi > self.mfi_upper)

            if buy_trigger and buy_guard:
                return Signal(action="buy", strength=1.0, confidence=0.8)
            elif sell_trigger and sell_guard:
                return Signal(action="sell", strength=1.0, confidence=0.8)
            else:
                return None
        else:
            return None