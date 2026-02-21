"""Fill simulation — realistic slippage model."""
import math


class FillSimulator:
    def __init__(self, slippage_pct: float = 0.05):
        self.slippage_pct = slippage_pct / 100

    def simulate_fill(
        self,
        bar_open: float,
        side: str,
        quantity: int,
        avg_volume: int = 1_000_000,
    ) -> float:
        """Returns fill price with slippage applied."""
        impact = self.slippage_pct
        # Extra impact for large orders (>1% of avg daily volume)
        if avg_volume > 0:
            volume_pct = quantity / avg_volume
            if volume_pct > 0.01:
                impact += math.sqrt(volume_pct) * 0.001

        if side in ("buy", "cover"):
            return bar_open * (1 + impact)
        else:  # sell, exit, short
            return bar_open * (1 - impact)
