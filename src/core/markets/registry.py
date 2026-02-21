"""
Market Registry — single source of truth for all market metadata.
Adding a new market = add one MarketConfig entry here. Zero other changes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from zoneinfo import ZoneInfo


class MarketCode(str, Enum):
    US = "US"   # NYSE / NASDAQ / BATS
    IN = "IN"   # NSE / BSE
    # Future: GB, JP, SG, AU — add here only


@dataclass(frozen=True)
class MarketConfig:
    code:                  MarketCode
    name:                  str
    exchange:              str
    currency:              str            # ISO 4217: "USD", "INR"
    currency_symbol:       str            # "$", "₹"
    timezone:              ZoneInfo
    session_open:          str            # "09:30" local time
    session_close:         str            # "16:00" local time
    settlement_days:       int            # T+ lag
    lot_size:              int            # minimum tradeable shares
    tick_size:             float          # minimum price increment
    default_universe:      str            # shown first in UniverseSelector
    data_providers:        list[str]      # preferred order; router tries each
    commission_model:      str            # maps to existing fill_model keys
    flags:                 dict = field(default_factory=dict)


MARKET_REGISTRY: dict[MarketCode, MarketConfig] = {

    MarketCode.US: MarketConfig(
        code=MarketCode.US,
        name="United States",
        exchange="NYSE/NASDAQ",
        currency="USD",
        currency_symbol="$",
        timezone=ZoneInfo("America/New_York"),
        session_open="09:30",
        session_close="16:00",
        settlement_days=2,
        lot_size=1,
        tick_size=0.01,
        default_universe="sp500_liquid",
        data_providers=["fmp", "alpaca"],
        commission_model="interactive_brokers",
    ),

    MarketCode.IN: MarketConfig(
        code=MarketCode.IN,
        name="India",
        exchange="NSE/BSE",
        currency="INR",
        currency_symbol="₹",
        timezone=ZoneInfo("Asia/Kolkata"),
        session_open="09:15",
        session_close="15:30",
        settlement_days=1,    # T+1 since Jan 2023
        lot_size=1,
        tick_size=0.05,
        default_universe="nifty50",
        data_providers=["nsepy", "upstox"],
        commission_model="zerodha_flat",   # ₹20 flat or 0.03% whichever lower
        flags={
            "stt_buy_pct":   0.001,   # 0.1% STT on delivery buy
            "stt_sell_pct":  0.001,   # 0.1% STT on delivery sell
            "gst_pct":       0.18,    # GST on brokerage
            "stamp_duty_pct": 0.00015, # 0.015% on buy value
            "has_fo_segment": True,
        },
    ),
}


def get_market(code: str) -> MarketConfig:
    try:
        return MARKET_REGISTRY[MarketCode(code.upper())]
    except (ValueError, KeyError):
        valid = [m.value for m in MARKET_REGISTRY]
        raise ValueError(f"Unknown market '{code}'. Valid: {valid}")


def list_markets() -> list[dict]:
    """Serialisable list for /api/v2/markets endpoint."""
    return [
        {
            "code":             m.code.value,
            "name":             m.name,
            "exchange":         m.exchange,
            "currency":         m.currency,
            "currency_symbol":  m.currency_symbol,
            "default_universe": m.default_universe,
            "is_default":       m.code == MarketCode.US,
        }
        for m in MARKET_REGISTRY.values()
    ]
