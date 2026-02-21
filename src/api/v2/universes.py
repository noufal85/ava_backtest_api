"""Universes endpoint — market-scoped universe definitions."""
from fastapi import APIRouter, Query

from src.core.markets.registry import get_market

router = APIRouter(tags=["Universes"])

UNIVERSE_DEFINITIONS: dict[str, list[dict]] = {
    "US": [
        {"name": "sp500", "description": "S&P 500", "type": "index", "symbol_count": 503},
        {"name": "sp500_liquid", "description": "S&P 500 Liquid (avg vol > 500k)", "type": "filter", "symbol_count": 400},
        {"name": "nasdaq100", "description": "NASDAQ 100", "type": "index", "symbol_count": 100},
        {"name": "russell2000", "description": "Russell 2000 Small Cap", "type": "index", "symbol_count": 2000},
    ],
    "IN": [
        {"name": "nifty50", "description": "Nifty 50", "type": "index", "symbol_count": 50},
        {"name": "nifty100", "description": "Nifty 100", "type": "index", "symbol_count": 100},
        {"name": "nifty500", "description": "Nifty 500", "type": "index", "symbol_count": 500},
        {"name": "nse_large_cap", "description": "NSE Large Cap", "type": "filter", "symbol_count": 100},
        {"name": "nse_mid_cap", "description": "NSE Mid Cap", "type": "filter", "symbol_count": 150},
        {"name": "nse_small_cap", "description": "NSE Small Cap", "type": "filter", "symbol_count": 250},
    ],
}


@router.get("/universes")
async def list_universes(market: str = Query("US")):
    """List defined universes for a market."""
    get_market(market)  # raises ValueError if invalid
    return UNIVERSE_DEFINITIONS.get(market.upper(), [])
