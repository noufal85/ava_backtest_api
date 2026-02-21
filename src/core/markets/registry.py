from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from zoneinfo import ZoneInfo

class MarketCode(str, Enum):
    US = "US"
    IN = "IN"

@dataclass(frozen=True)
class MarketConfig:
    code: MarketCode
    name: str
    exchange: str
    currency: str
    currency_symbol: str
    timezone: ZoneInfo
    session_open: str
    session_close: str
    settlement_days: int
    lot_size: int
    tick_size: float
    default_universe: str
    data_providers: list[str]
    commission_model: str
    flags: dict = field(default_factory=dict)

MARKET_REGISTRY: dict[MarketCode, MarketConfig] = {
    MarketCode.US: MarketConfig(
        code=MarketCode.US, name="United States", exchange="NYSE/NASDAQ",
        currency="USD", currency_symbol="$",
        timezone=ZoneInfo("America/New_York"),
        session_open="09:30", session_close="16:00",
        settlement_days=2, lot_size=1, tick_size=0.01,
        default_universe="sp500_liquid",
        data_providers=["fmp", "alpaca"],
        commission_model="interactive_brokers",
    ),
    MarketCode.IN: MarketConfig(
        code=MarketCode.IN, name="India", exchange="NSE/BSE",
        currency="INR", currency_symbol="₹",
        timezone=ZoneInfo("Asia/Kolkata"),
        session_open="09:15", session_close="15:30",
        settlement_days=1, lot_size=1, tick_size=0.05,
        default_universe="nifty50",
        data_providers=["nsepy", "upstox"],
        commission_model="zerodha_flat",
        flags={"stt_buy_pct": 0.001, "stt_sell_pct": 0.001,
               "gst_pct": 0.18, "stamp_duty_pct": 0.00015},
    ),
}

def get_market(code: str) -> MarketConfig:
    try:
        return MARKET_REGISTRY[MarketCode(code.upper())]
    except (ValueError, KeyError):
        raise ValueError(f"Unknown market {code!r}. Valid: {[m.value for m in MARKET_REGISTRY]}")

def list_markets() -> list[dict]:
    return [{"code": m.code.value, "name": m.name, "exchange": m.exchange,
             "currency": m.currency, "currency_symbol": m.currency_symbol,
             "default_universe": m.default_universe,
             "is_default": m.code == MarketCode.US}
            for m in MARKET_REGISTRY.values()]
