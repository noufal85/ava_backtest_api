"""Strategy 6: RSI + Volatility Filter — regime-aware mean reversion."""
import polars as pl

from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register


def _compute_rsi_wilder(closes: list[float], period: int) -> list[float]:
    """Compute RSI using Wilder smoothing."""
    if len(closes) < period + 1:
        return []

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    gains = [max(d, 0.0) for d in deltas[:period]]
    losses = [max(-d, 0.0) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    rsi_values: list[float] = []
    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100.0 - 100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        d = deltas[i]
        gain = max(d, 0.0)
        loss = max(-d, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - 100.0 / (1.0 + rs))

    return rsi_values


def _compute_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float]:
    """Compute ATR series using Wilder smoothing."""
    if len(closes) < period + 1:
        return []

    # True range: max(high-low, |high-prev_close|, |low-prev_close|)
    tr_values = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_values.append(max(hl, hc, lc))

    if len(tr_values) < period:
        return []

    # First ATR = simple average of first `period` TR values
    atr_values = [sum(tr_values[:period]) / period]

    # Wilder smoothing
    for i in range(period, len(tr_values)):
        atr_values.append((atr_values[-1] * (period - 1) + tr_values[i]) / period)

    return atr_values


def _percentile_rank(values: list[float], current: float) -> float:
    """Percentile rank of current value within the values list (0-100)."""
    if not values:
        return 50.0
    count_below = sum(1 for v in values if v < current)
    return (count_below / len(values)) * 100.0


@register
class RSIVolFilter(BaseStrategy):
    name = "rsi_vol_filter"
    version = "2.0.0"
    description = "RSI mean reversion gated by ATR volatility percentile rank."
    category = "multi_factor"
    tags = ["mean_reversion", "volatility", "regime_aware", "advanced"]

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        atr_period: int = 14,
        atr_rank_period: int = 100,
        vol_threshold: float = 50.0,
        vol_exit_threshold: float = 80.0,
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.atr_period = atr_period
        self.atr_rank_period = atr_rank_period
        self.vol_threshold = vol_threshold
        self.vol_exit_threshold = vol_exit_threshold
        self._in_position: bool = False

    def get_parameter_schema(self) -> dict:
        return {
            "rsi_period":         {"type": "integer", "default": 14,   "minimum": 5,   "maximum": 50,  "description": "RSI period"},
            "oversold":           {"type": "number",  "default": 30.0, "minimum": 10,  "maximum": 45,  "description": "RSI oversold level"},
            "overbought":         {"type": "number",  "default": 70.0, "minimum": 55,  "maximum": 90,  "description": "RSI overbought level"},
            "atr_period":         {"type": "integer", "default": 14,   "minimum": 5,   "maximum": 50,  "description": "ATR volatility period"},
            "atr_rank_period":    {"type": "integer", "default": 100,  "minimum": 20,  "maximum": 252, "description": "ATR rank lookback period"},
            "vol_threshold":      {"type": "number",  "default": 50.0, "minimum": 0,   "maximum": 100, "description": "Min ATR percentile rank to enter"},
            "vol_exit_threshold": {"type": "number",  "default": 80.0, "minimum": 0,   "maximum": 100, "description": "ATR percentile rank for vol exit"},
        }

    def get_warmup_periods(self) -> int:
        return max(self.rsi_period + 2, self.atr_period + self.atr_rank_period + 1)

    def generate_signal(self, window) -> Signal | None:
        hist = window.historical()
        cur = window.current_bar()
        combined = pl.concat([hist, cur])

        closes = combined["close"].to_list()
        highs = combined["high"].to_list()
        lows = combined["low"].to_list()

        rsi_values = _compute_rsi_wilder(closes, self.rsi_period)
        atr_values = _compute_atr(highs, lows, closes, self.atr_period)

        if len(rsi_values) < 2 or len(atr_values) < self.atr_rank_period:
            return None

        rsi_now = rsi_values[-1]
        atr_now = atr_values[-1]
        atr_rank = _percentile_rank(atr_values[-self.atr_rank_period :], atr_now)

        if self._in_position:
            # Exit: RSI > overbought OR ATR percentile rank > vol_exit_threshold
            if rsi_now > self.overbought or atr_rank > self.vol_exit_threshold:
                self._in_position = False
                reason = "overbought" if rsi_now > self.overbought else "high_vol"
                return Signal(
                    action="sell",
                    strength=1.0,
                    confidence=0.7,
                    metadata={"rsi": rsi_now, "atr": atr_now, "atr_rank": atr_rank, "exit_reason": reason},
                )
        else:
            # Entry: RSI < oversold AND ATR percentile rank < vol_threshold
            if rsi_now < self.oversold and atr_rank < self.vol_threshold:
                self._in_position = True
                return Signal(
                    action="buy",
                    strength=1.0,
                    confidence=0.85,
                    metadata={"rsi": rsi_now, "atr": atr_now, "atr_rank": atr_rank},
                )

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "rsi_period": {"type": "integer", "default": 14, "min": 5, "max": 50, "step": 1},
            "oversold": {"type": "number", "default": 30, "min": 10, "max": 40, "step": 5},
            "overbought": {"type": "number", "default": 70, "min": 60, "max": 90, "step": 5},
            "atr_period": {"type": "integer", "default": 14, "min": 5, "max": 30, "step": 1},
            "atr_rank_period": {"type": "integer", "default": 100, "min": 20, "max": 252, "step": 10},
            "vol_threshold": {"type": "number", "default": 50, "min": 20, "max": 80, "step": 5},
            "vol_exit_threshold": {"type": "number", "default": 80, "min": 50, "max": 95, "step": 5},
        }
