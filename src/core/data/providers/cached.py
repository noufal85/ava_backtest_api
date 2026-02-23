"""CachedProvider — wraps any DataProvider with FileCache for local-first data access."""
from datetime import date

import polars as pl
import structlog

from src.core.data.cache.file_cache import FileCache
from src.core.data.providers.base import DataProvider
from src.core.markets.registry import MarketCode

logger = structlog.get_logger()


class CachedProvider(DataProvider):
    """Decorator that checks FileCache before hitting the upstream provider."""

    def __init__(self, upstream: DataProvider, cache: FileCache | None = None, max_age_hours: float = 24):
        self._upstream = upstream
        self._cache = cache or FileCache()
        self._max_age_hours = max_age_hours

    @property
    def name(self) -> str:
        return f"cached_{self._upstream.name}"

    @property
    def supported_markets(self) -> list[MarketCode]:
        return self._upstream.supported_markets

    async def fetch_ohlcv(
        self,
        symbol: str,
        market: MarketCode,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        market_str = market.value if hasattr(market, "value") else str(market)

        # Check cache first
        if not self._cache.is_stale(market_str, symbol, start, end, timeframe, self._max_age_hours):
            cached = self._cache.get(market_str, symbol, start, end, timeframe)
            if cached is not None and not cached.is_empty():
                logger.info("cache.hit", symbol=symbol, rows=len(cached))
                return cached

        # Miss — fetch from upstream
        logger.info("cache.miss", symbol=symbol, market=market_str)
        df = await self._upstream.fetch_ohlcv(symbol, market, start, end, timeframe)

        # Store in cache
        if not df.is_empty():
            self._cache.put(df, market_str, symbol, start, end, timeframe)
            logger.info("cache.stored", symbol=symbol, rows=len(df))

        return df

    async def search_symbols(self, query: str, market: MarketCode) -> list[dict]:
        return await self._upstream.search_symbols(query, market)

    async def get_universe_symbols(self, universe_name: str, market: MarketCode) -> list[str]:
        return await self._upstream.get_universe_symbols(universe_name, market)
