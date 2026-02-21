"""Universe manager — resolves universe names to symbol lists."""
from src.core.data.providers.router import ProviderRouter
from src.core.markets.registry import MarketCode


class UniverseManager:

    def __init__(self, router: ProviderRouter):
        self.router = router
        self._cache: dict[tuple, list[str]] = {}

    async def get_symbols(self, universe_name: str, market: MarketCode) -> list[str]:
        key = (universe_name, market.value)
        if key not in self._cache:
            self._cache[key] = await self.router.get_universe_symbols(universe_name, market)
        return self._cache[key]
