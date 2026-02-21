"""Data endpoints — candles and symbol search."""
from fastapi import APIRouter, Query

from src.core.markets.registry import get_market

router = APIRouter(tags=["Data"])


@router.get("/data/candles")
async def get_candles(
    symbol: str = Query(...),
    market: str = Query("US"),
    timeframe: str = Query("1d"),
    start: str = Query(...),
    end: str = Query(...),
):
    """Query candle data."""
    get_market(market)
    # Router + cache wired in EP-8 DB integration
    return {"symbol": symbol, "market": market, "timeframe": timeframe, "candles": []}


@router.get("/symbols/search")
async def search_symbols(q: str = Query(...), market: str = Query("US")):
    """Search symbols by query string."""
    get_market(market)
    return {"results": [], "market": market}
