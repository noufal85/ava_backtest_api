from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

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
class TvRmi3Strategy(BaseStrategy):
    name = "tv_rmi_3_strategy"
    version = "1.0.0"
    description = "RMI-3 Strategy: Triple Relative Momentum Index momentum strategy"
    category = "multi_factor"
    tags = ["momentum", "mean_reversion"]

    def __init__(
        self,
        resolution: int = 3,
        alpha_period: int = 29,
        alpha_momentum: int = 7,
        beta_period: int = 239,
        beta_momentum: int = 23,
        gamma_period: int = 2467,
        gamma_momentum: int = 239,
        alpha_high_boundary: int = 23,
        alpha_low_boundary: int = -23,
        beta_high_boundary: int = 19,
        beta_low_boundary: int = -19,
        gamma_high_boundary: int = 19,
        gamma_low_boundary: int = -19,
        take_profit_pct: float = 0.35,
        stop_loss_pct: float = 0.30,
    ):
        self.resolution = resolution
        self.alpha_period = alpha_period
        self.alpha_momentum = alpha_momentum
        self.beta_period = beta_period
        self.beta_momentum = beta_momentum
        self.gamma_period = gamma_period
        self.gamma_momentum = gamma_momentum
        self.alpha_high_boundary = alpha_high_boundary
        self.alpha_low_boundary = alpha_low_boundary
        self.beta_high_boundary = beta_high_boundary
        self.beta_low_boundary = beta_low_boundary
        self.gamma_high_boundary = gamma_high_boundary
        self.gamma_low_boundary = gamma_low_boundary
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct

        self.alpha_period_scaled = max(self.alpha_period // self.resolution, 1)
        self.alpha_momentum_scaled = max(self.alpha_momentum // self.resolution, 1)
        self.beta_period_scaled = max(self.beta_period // self.resolution, 1)
        self.beta_momentum_scaled = max(self.beta_momentum // self.resolution, 1)
        self.gamma_period_scaled = max(self.gamma_period // self.resolution, 1)
        self.gamma_momentum_scaled = max(self.gamma_momentum // self.resolution, 1)

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "resolution": {"type": "integer", "default": 3},
                "alpha_period": {"type": "integer", "default": 29},
                "alpha_momentum": {"type": "integer", "default": 7},
                "beta_period": {"type": "integer", "default": 239},
                "beta_momentum": {"type": "integer", "default": 23},
                "gamma_period": {"type": "integer", "default": 2467},
                "gamma_momentum": {"type": "integer", "default": 239},
                "alpha_high_boundary": {"type": "integer", "default": 23},
                "alpha_low_boundary": {"type": "integer", "default": -23},
                "beta_high_boundary": {"type": "integer", "default": 19},
                "beta_low_boundary": {"type": "integer", "default": -19},
                "gamma_high_boundary": {"type": "integer", "default": 19},
                "gamma_low_boundary": {"type": "integer", "default": -19},
                "take_profit_pct": {"type": "number", "default": 0.35},
                "stop_loss_pct": {"type": "number", "default": 0.30},
            },
        }

    def get_warmup_periods(self) -> int:
        return max(self.gamma_period_scaled + self.gamma_momentum_scaled, 500) + 50 # Buffer

    def _rmi(self, closes: list[float], length: int, momentum: int) -> list[float]:
        """Calculate Relative Momentum Index (RMI)."""
        rmi_values = []
        up_ema_values = []
        down_ema_values = []

        for i in range(len(closes)):
            if i < momentum:
                up_ema_values.append(0.0)
                down_ema_values.append(0.0)
                rmi_values.append(0.0)
                continue

            up_move = max(closes[i] - closes[i - momentum], 0)
            down_move = max(closes[i - momentum] - closes[i], 0)

            if i == momentum:
                up_ema = sum([max(closes[j] - closes[j - momentum], 0) for j in range(momentum, i + 1)]) / length
                down_ema = sum([max(closes[j - momentum] - closes[j], 0) for j in range(momentum, i + 1)]) / length
            else:
                up_ema = (up_move * (2 / (length + 1))) + (up_ema_values[-1] * (1 - (2 / (length + 1))))
                down_ema = (down_move * (2 / (length + 1))) + (down_ema_values[-1] * (1 - (2 / (length + 1))))

            up_ema_values.append(up_ema)
            down_ema_values.append(down_ema)

            if down_ema == 0:
                rmi = 50
            else:
                rmi = 50 - 100 / (1 + up_ema / down_ema)
            rmi_values.append(rmi)

        return rmi_values

    def generate_signal(self, window) -> Optional[Signal]:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        closes = df["close"].to_list()

        rmi_alpha_values = self._rmi(closes, self.alpha_period_scaled, self.alpha_momentum_scaled)
        rmi_beta_values = self._rmi(closes, self.beta_period_scaled, self.beta_momentum_scaled)
        rmi_gamma_values = self._rmi(closes, self.gamma_period_scaled, self.gamma_momentum_scaled)

        alpha_rmi = rmi_alpha_values[-1]
        beta_rmi = rmi_beta_values[-1]
        gamma_rmi = rmi_gamma_values[-1]

        alpha_rmi_prev = rmi_alpha_values[-2] if len(rmi_alpha_values) > 1 else None
        beta_rmi_prev = rmi_beta_values[-2] if len(rmi_beta_values) > 1 else None
        gamma_rmi_prev = rmi_gamma_values[-2] if len(rmi_gamma_values) > 1 else None

        if alpha_rmi_prev is None or beta_rmi_prev is None or gamma_rmi_prev is None:
            return None

        alpha_dip = alpha_rmi_prev <= self.alpha_low_boundary
        beta_dip = beta_rmi_prev <= self.beta_low_boundary
        gamma_dip = gamma_rmi_prev <= self.gamma_low_boundary

        alpha_top = alpha_rmi_prev >= self.alpha_high_boundary
        beta_top = beta_rmi_prev >= self.beta_high_boundary
        gamma_top = gamma_rmi_prev >= self.gamma_high_boundary

        rmi_dip_3 = alpha_dip and beta_dip and gamma_dip
        rmi_top_3 = alpha_top and beta_top and gamma_top

        if rmi_dip_3:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"rmi_alpha": alpha_rmi, "rmi_beta": beta_rmi, "rmi_gamma": gamma_rmi})
        elif rmi_top_3:
            return Signal(action="sell", strength=1.0, confidence=1.0, metadata={"rmi_alpha": alpha_rmi, "rmi_beta": beta_rmi, "rmi_gamma": gamma_rmi})
        else:
            return None