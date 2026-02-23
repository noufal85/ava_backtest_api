"""sector_rotation — converted from trading-backtester."""
import polars as pl
import numpy as np
import math
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register

@register
class SectorRotation(BaseStrategy):
    name = "sector_rotation"
    name: str = "sector_rotation"
    version: str = "1.0.0"
    description: str = "Momentum: buy top-performing symbols by rolling return percentile"
    category: str = "momentum"
    tags: list[str] = ["momentum", "sector rotation"]

    def __init__(self, lookback: int = 63, ranking_period: int = 252, percentile_threshold: int = 75, hold_days: int = 21):
        self.lookback = lookback
        self.ranking_period = ranking_period
        self.percentile_threshold = percentile_threshold
        self.hold_days = hold_days

    def get_warmup_periods(self) -> int:
        return self.ranking_period + self.lookback + 1

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()

        if len(closes) < self.ranking_period + self.lookback:
            return None

        n_day_ret_list = n_day_return(closes, self.lookback)
        n_day_ret = n_day_ret_list[-1]

        if len(n_day_ret_list) < self.ranking_period:
            return None

        ret_window = n_day_ret_list[-(self.ranking_period):-1]

        if not ret_window:
            return None

        ret_percentile = np.percentile(ret_window, self.percentile_threshold)
        ret_percentile_25 = np.percentile(ret_window, 25)

        if n_day_ret > 0 and n_day_ret > ret_percentile:
            return Signal(action="buy", strength=min(1.0, n_day_ret), confidence=1.0, metadata={"n_day_ret": n_day_ret})
        elif n_day_ret < ret_percentile_25:
            return Signal(action="sell", strength=min(1.0, abs(n_day_ret)), confidence=1.0, metadata={"n_day_ret": n_day_ret})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "lookback": {"type": "integer", "default": 63, "description": "Return calculation period"},
            "ranking_period": {"type": "integer", "default": 252, "description": "Rolling window for percentile calc"},
            "percentile_threshold": {"type": "integer", "default": 75, "description": "Minimum percentile to trigger buy"},
            "hold_days": {"type": "integer", "default": 21, "description": "Fixed holding period"},
        }