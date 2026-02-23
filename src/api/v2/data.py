"""Data endpoints — candles and symbol search (cache-first)."""
from datetime import date

from fastapi import APIRouter, Query

from src.core.data import get_provider
from src.core.markets.registry import MarketCode, get_market

router = APIRouter(tags=["Data"])


@router.get("/data/candles")
async def get_candles(
    symbol: str = Query(...),
    market: str = Query("US"),
    timeframe: str = Query("1d"),
    start: str = Query(...),
    end: str = Query(...),
):
    """Query candle data — served from local cache when available."""
    get_market(market)
    provider = get_provider()
    market_code = MarketCode.US if market == "US" else MarketCode.IN
    df = await provider.fetch_ohlcv(
        symbol, market_code, date.fromisoformat(start), date.fromisoformat(end), timeframe
    )
    candles = df.to_dicts() if not df.is_empty() else []
    # Serialize datetimes to strings
    for c in candles:
        if "ts" in c:
            c["ts"] = str(c["ts"])[:10]
    return {"symbol": symbol, "market": market, "timeframe": timeframe, "candles": candles}


@router.get("/symbols/search")
async def search_symbols(q: str = Query(...), market: str = Query("US")):
    """Search symbols by query string."""
    get_market(market)
    provider = get_provider()
    market_code = MarketCode.US if market == "US" else MarketCode.IN
    results = await provider.search_symbols(q, market_code)
    return {"results": results, "market": market}
