from abc import ABC, abstractmethod
from datetime import date
import polars as pl
from src.core.markets.registry import MarketCode

class DataProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def supported_markets(self) -> list[MarketCode]: ...
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, market: MarketCode, start: date, end: date, timeframe: str = "1d") -> pl.DataFrame: ...
    @abstractmethod
    async def search_symbols(self, query: str, market: MarketCode) -> list[dict]: ...
    @abstractmethod
    async def get_universe_symbols(self, universe_name: str, market: MarketCode) -> list[str]: ...
