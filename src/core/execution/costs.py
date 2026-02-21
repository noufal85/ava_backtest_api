"""Market-aware cost model — commissions + India STT/GST/stamp duty."""
from src.core.markets.registry import MarketCode, MARKET_REGISTRY


class InteractiveBrokersCommission:
    """$0.005 per share, minimum $1.00 per order."""

    def calculate(self, quantity: int, price: float, side: str) -> float:
        return max(1.0, quantity * 0.005)


class ZerodhaFlatCommission:
    """₹20 flat or 0.03% of trade value, whichever is lower."""

    def calculate(self, quantity: int, price: float, side: str) -> float:
        return min(20.0, quantity * price * 0.0003)


class MarketAwareCostModel:
    def calculate_total(
        self,
        quantity: int,
        price: float,
        side: str,
        market: MarketCode,
    ) -> dict[str, float]:
        config = MARKET_REGISTRY[market]
        trade_value = quantity * price

        if config.commission_model == "interactive_brokers":
            commission = InteractiveBrokersCommission().calculate(quantity, price, side)
        elif config.commission_model == "zerodha_flat":
            commission = ZerodhaFlatCommission().calculate(quantity, price, side)
        else:
            commission = 0.0

        stt = gst = stamp_duty = 0.0
        if market == MarketCode.IN:
            flags = config.flags
            stt_pct = flags.get(
                "stt_buy_pct" if side in ("buy", "cover") else "stt_sell_pct", 0
            )
            stt = trade_value * stt_pct
            gst = commission * flags.get("gst_pct", 0)
            if side in ("buy", "cover"):
                stamp_duty = trade_value * flags.get("stamp_duty_pct", 0)

        total = commission + stt + gst + stamp_duty
        return {
            "commission": commission,
            "stt": stt,
            "gst": gst,
            "stamp_duty": stamp_duty,
            "total": total,
        }
