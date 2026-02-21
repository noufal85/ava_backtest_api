"""ProviderRouter — selects and fails-over between DataProviders for a market."""
from datetime import date

import polars as pl
import structlog

from src.core.markets.registry import MarketCode, MARKET_REGISTRY
from src.core.data.providers.base import DataProvider

logger = structlog.get_logger()


class ProviderRouter:

    def __init__(self, providers: list[DataProvider]):
        # Index: (market, provider_name) -> provider
        self._index: dict[tuple[MarketCode, str], DataProvider] = {
            (m, p.name): p
            for p in providers
            for m in p.supported_markets
        }

    async def fetch_ohlcv(
        self,
        symbol: str,
        market: MarketCode,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        config = MARKET_REGISTRY[market]
        last_err = None
        for pname in config.data_providers:
            key = (market, pname)
            if key not in self._index:
                continue
            try:
                df = await self._index[key].fetch_ohlcv(symbol, market, start, end, timeframe)
                logger.info("provider.ok", provider=pname, symbol=symbol, rows=len(df))
                return df
            except Exception as e:
                logger.warning("provider.failed", provider=pname, error=str(e))
                last_err = e
        raise RuntimeError(f"All providers failed for {symbol}@{market}: {last_err}")

    async def search_symbols(self, query: str, market: MarketCode) -> list[dict]:
        config = MARKET_REGISTRY[market]
        for pname in config.data_providers:
            if (market, pname) in self._index:
                return await self._index[(market, pname)].search_symbols(query, market)
        return []

    async def get_universe_symbols(self, universe_name: str, market: MarketCode) -> list[str]:
        config = MARKET_REGISTRY[market]
        for pname in config.data_providers:
            if (market, pname) in self._index:
                return await self._index[(market, pname)].get_universe_symbols(universe_name, market)
        raise RuntimeError(f"No provider for universe {universe_name!r} in {market}")
