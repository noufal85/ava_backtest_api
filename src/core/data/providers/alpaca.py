"""Alpaca provider — US equities fallback via Alpaca Data API v2."""
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

import aiohttp
import polars as pl

from src.core.data.providers.base import DataProvider
from src.core.markets.registry import MarketCode

ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"
ALPACA_API_BASE = "https://api.alpaca.markets/v2"
ET = ZoneInfo("America/New_York")

# Same static universe lists as FMP for consistency
SP500_SAMPLE = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AIG",
    "AMAT", "AMD", "AMGN", "AMZN", "AVGO", "AXP", "BA", "BAC", "BK", "BKNG",
    "BLK", "BMY", "BRK.B", "C", "CAT", "CHTR", "CL", "CMCSA", "COF", "COP",
    "COST", "CRM", "CSCO", "CVX", "D", "DE", "DHR", "DIS", "DOW", "DUK",
    "EMR", "EXC", "F", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL",
    "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KHC",
    "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET",
    "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA",
    "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW",
    "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP",
    "UPS", "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM",
]

NASDAQ100_SAMPLE = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "ALGN", "AMAT", "AMD",
    "AMGN", "AMZN", "ANSS", "ASML", "ATVI", "AVGO", "AZN", "BIIB", "BKNG", "BKR",
    "CDNS", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSGP", "CSX",
    "CTAS", "CTSH", "DDOG", "DLTR", "DXCM", "EA", "EBAY", "ENPH", "EXC", "FANG",
    "FAST", "FTNT", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "ILMN", "INTC", "INTU",
    "ISRG", "JD", "KDP", "KHC", "KLAC", "LCID", "LRCX", "LULU", "MAR", "MCHP",
    "MDLZ", "MELI", "META", "MNST", "MRNA", "MRVL", "MSFT", "MU", "NFLX", "NVDA",
    "NXPI", "ODFL", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PYPL", "QCOM",
    "REGN", "RIVN", "ROST", "SBUX", "SIRI", "SNPS", "TEAM", "TMUS", "TSLA", "TXN",
    "VRSK", "VRTX", "WBA", "WBD", "WDAY", "XEL", "ZM", "ZS",
]


class AlpacaProvider(DataProvider):

    def __init__(self, api_key: str | None = None, secret_key: str | None = None):
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")

    @property
    def name(self) -> str:
        return "alpaca"

    @property
    def supported_markets(self) -> list[MarketCode]:
        return [MarketCode.US]

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
        }

    async def fetch_ohlcv(
        self,
        symbol: str,
        market: MarketCode,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        if market != MarketCode.US:
            raise ValueError(f"Alpaca only supports US market, got {market}")

        tf_map = {"1d": "1Day", "1h": "1Hour", "15m": "15Min", "5m": "5Min", "1m": "1Min"}
        alpaca_tf = tf_map.get(timeframe, "1Day")

        url = f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars"
        params = {
            "timeframe": alpaca_tf,
            "start": f"{start.isoformat()}T00:00:00Z",
            "end": f"{end.isoformat()}T23:59:59Z",
            "limit": 10000,
            "adjustment": "all",
        }

        all_bars = []
        async with aiohttp.ClientSession(headers=self._headers()) as session:
            while True:
                async with session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                bars = data.get("bars", [])
                if not bars:
                    break
                all_bars.extend(bars)

                next_token = data.get("next_page_token")
                if not next_token:
                    break
                params["page_token"] = next_token

        if not all_bars:
            raise ValueError(f"No data returned for {symbol} from Alpaca")

        df = pl.DataFrame(all_bars)

        # Alpaca bar fields: t, o, h, l, c, v, n, vw
        df = df.rename({
            "t": "ts",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })

        # Parse timestamp
        df = df.with_columns(
            pl.col("ts")
            .str.to_datetime()
            .dt.convert_time_zone("America/New_York")
        )

        # adj_close = close for Alpaca (already adjusted when adjustment=all)
        df = df.with_columns(pl.col("close").alias("adj_close"))

        return (
            df.select(["ts", "open", "high", "low", "close", "volume", "adj_close"])
            .sort("ts")
        )

    async def search_symbols(self, query: str, market: MarketCode) -> list[dict]:
        url = f"{ALPACA_API_BASE}/assets"
        params = {
            "status": "active",
            "asset_class": "us_equity",
        }

        async with aiohttp.ClientSession(headers=self._headers()) as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                assets = await resp.json()

        q = query.upper()
        matches = [
            {
                "symbol": a.get("symbol", ""),
                "name": a.get("name", ""),
                "exchange": a.get("exchange", ""),
                "sector": "",
            }
            for a in assets
            if q in a.get("symbol", "").upper() or q in a.get("name", "").upper()
        ]
        return matches[:20]

    async def get_universe_symbols(self, universe_name: str, market: MarketCode) -> list[str]:
        universes = {
            "sp500": SP500_SAMPLE,
            "sp500_liquid": SP500_SAMPLE,
            "nasdaq100": NASDAQ100_SAMPLE,
        }
        if universe_name not in universes:
            raise ValueError(f"Unknown US universe: {universe_name!r}")
        return universes[universe_name]
