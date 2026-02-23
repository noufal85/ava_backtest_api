"""volume_spike — converted from trading-backtester."""
import polars as pl
import numpy as np
import math
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register

@register
class VolumeSpike(BaseStrategy):
    name = "volume_spike"
    name: str = "volume_spike"
    version: str = "1.1.0"
    description: str = "Volume Spike Momentum: detect institutional buying/selling via volume spikes (long & short)"
    category: str = "volatility"
    tags: list[str] = ["volume", "momentum"]

    def __init__(self, volume_mult: float = 2.0, min_move_pct: float = 0.02, hold_days: int = 5):
        self.volume_mult = volume_mult
        self.min_move_pct = min_move_pct
        self.hold_days = hold_days

    def get_warmup_periods(self) -> int:
        return 20 + 5  # 20 for volume_sma, 5 buffer

    def generate_signal(self, window) -> Signal | None:
        historical_data = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_data, current_bar])

        if len(df) < 21:
            return None

        volumes = df["volume"].to_list()
        closes = df["close"].to_list()

        vol_sma_20 = volume_sma(volumes, 20)
        price_change_pct = [(closes[i] - closes[i-1]) / closes[i-1] if i > 0 else 0.0 for i in range(len(closes))]

        prev_volume = volumes[-2]
        prev_vol_sma = vol_sma_20[-2] if len(vol_sma_20) > 1 else np.nan
        prev_price_chg = price_change_pct[-2] if len(price_change_pct) > 1 else 0.0

        if np.isnan(prev_vol_sma):
            return None

        vol_spike = prev_volume > self.volume_mult * prev_vol_sma

        # Long: volume spike + positive price move
        price_move_up = prev_price_chg > self.min_move_pct
        if vol_spike and price_move_up:
            return Signal(action="buy", strength=min(prev_price_chg / self.min_move_pct, 1.0), confidence=1.0, metadata={"price_change_pct": prev_price_chg})

        # Short: volume spike + negative price move
        price_move_down = prev_price_chg < -self.min_move_pct
        if vol_spike and price_move_down:
            return Signal(action="sell", strength=min(abs(prev_price_chg) / self.min_move_pct, 1.0), confidence=1.0, metadata={"price_change_pct": prev_price_chg})

        return None

    def get_parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "volume_mult": {
                    "type": "number",
                    "default": 2.0,
                    "description": "Volume spike multiplier"
                },
                "min_move_pct": {
                    "type": "number",
                    "default": 0.02,
                    "description": "Minimum price move percentage"
                },
                "hold_days": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum holding period in trading days"
                }
            },
            "required": ["volume_mult", "min_move_pct", "hold_days"]
        }