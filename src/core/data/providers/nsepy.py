"""NSEpy provider — free NSE/BSE daily data."""
import asyncio
from datetime import date

import polars as pl

from src.core.data.providers.base import DataProvider
from src.core.markets.registry import MarketCode

NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BAJFINANCE", "BHARTIARTL", "KOTAKBANK", "LT", "ASIANPAINT", "AXISBANK",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "POWERGRID",
    "NTPC", "TECHM", "HCLTECH", "ONGC", "COALINDIA", "BAJAJFINSV", "INDUSINDBK",
    "GRASIM", "CIPLA", "BRITANNIA", "DIVISLAB", "DRREDDY", "EICHERMOT", "BPCL",
    "HEROMOTOCO", "HINDALCO", "JSWSTEEL", "M&M", "SHREECEM", "TATACONSUM",
    "TATAMOTORS", "TATASTEEL", "APOLLOHOSP", "ADANIPORTS", "SBILIFE",
    "HDFCLIFE", "UPL", "BAJAJ-AUTO", "ITC", "ADANIENT",
]

UNIVERSES = {
    "nifty50": NIFTY50,
    "nifty100": NIFTY50,
    "nifty500": NIFTY50,
    "nse_large_cap": NIFTY50,
    "nse_mid_cap": [],
    "nse_small_cap": [],
}


class NSEpyProvider(DataProvider):

    @property
    def name(self) -> str:
        return "nsepy"

    @property
    def supported_markets(self) -> list[MarketCode]:
        return [MarketCode.IN]

    async def fetch_ohlcv(
        self,
        symbol: str,
        market: MarketCode,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        if timeframe != "1d":
            raise ValueError(f"NSEpy supports daily only. Got: {timeframe!r}")
        loop = asyncio.get_event_loop()
        df_pd = await loop.run_in_executor(None, self._fetch, symbol, start, end)
        return self._normalise(df_pd)

    def _fetch(self, symbol: str, start: date, end: date):
        from nsepy import get_history
        clean = symbol.replace(".NS", "").replace(".BO", "").upper()
        return get_history(symbol=clean, start=start, end=end)

    def _normalise(self, df_pd) -> pl.DataFrame:
        df = pl.from_pandas(df_pd.reset_index())
        renames = {
            "Date": "ts",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = df.rename({k: v for k, v in renames.items() if k in df.columns})
        if "adj_close" not in df.columns:
            df = df.with_columns(pl.col("close").alias("adj_close"))
        return df.select(["ts", "open", "high", "low", "close", "volume", "adj_close"]).sort("ts")

    async def search_symbols(self, query: str, market: MarketCode) -> list[dict]:
        q = query.upper()
        return [
            {"symbol": s + ".NS", "name": s, "exchange": "NSE", "sector": ""}
            for s in NIFTY50
            if q in s
        ]

    async def get_universe_symbols(self, universe_name: str, market: MarketCode) -> list[str]:
        if universe_name not in UNIVERSES:
            raise ValueError(f"Unknown India universe: {universe_name!r}")
        return [s + ".NS" for s in UNIVERSES[universe_name]]
