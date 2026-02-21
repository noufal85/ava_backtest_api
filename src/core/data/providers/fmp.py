"""FMP (Financial Modeling Prep) provider — US equities."""
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

import aiohttp
import polars as pl
from aiolimiter import AsyncLimiter

from src.core.data.providers.base import DataProvider
from src.core.markets.registry import MarketCode

FMP_BASE = "https://financialmodelingprep.com/api/v3"
ET = ZoneInfo("America/New_York")

# Static universe lists
SP500_ENDPOINT = "/sp500_constituent"
NASDAQ100_ENDPOINT = "/nasdaq_constituent"

# Russell 2000 — first 100 representative symbols
RUSSELL2000_SAMPLE = [
    "ACIW","ACHC","AEIS","AGEN","AGYS","AIMC","ALGT","AMED","AMPH","ANET",
    "APPF","ARLP","AROC","ARWR","ATKR","AVAV","AXNX","AYI","BBIO","BCPC",
    "BDC","BFAM","BGS","BHF","BILL","BJ","BKH","BLKB","BMRN","BOOT",
    "BRO","CAKE","CALM","CASA","CASY","CBRL","CBT","CCOI","CDW","CEIX",
    "CGNX","CHDN","CHE","CIEN","CLS","CMC","CNK","CNMD","COHR","COLB",
    "CORT","CPRT","CRI","CRVL","CSWI","CVBF","CW","CWST","CYTK","DAN",
    "DORM","EAT","EBC","EFSC","EHC","ENSG","EPAM","ESGR","ESNT","ETSY",
    "EVR","EXLS","EXP","FFIN","FIVE","FIX","FLO","FNB","FOXF","FTDR",
    "FULT","G","GEF","GKOS","GLOB","GOLF","GSHD","GTY","HAE","HALO",
    "HAYW","HBI","HCI","HELE","HLI","HLNE","HNI","HP","HQY","HURN",
]


class FMPProvider(DataProvider):

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("FMP_API_KEY", "")
        self._limiter = AsyncLimiter(10, 60)  # 10 requests per minute (free tier)

    @property
    def name(self) -> str:
        return "fmp"

    @property
    def supported_markets(self) -> list[MarketCode]:
        return [MarketCode.US]

    async def fetch_ohlcv(
        self,
        symbol: str,
        market: MarketCode,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        if market != MarketCode.US:
            raise ValueError(f"FMP only supports US market, got {market}")

        url = f"{FMP_BASE}/historical-price-full/{symbol}"
        params = {
            "from": start.isoformat(),
            "to": end.isoformat(),
            "apikey": self._api_key,
        }

        async with self._limiter:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

        historical = data.get("historical", [])
        if not historical:
            raise ValueError(f"No data returned for {symbol} from FMP")

        df = pl.DataFrame(historical)

        # Rename and select standard columns
        df = df.rename({
            "date": "ts",
            "adjClose": "adj_close",
        })

        # Parse ts as datetime with ET timezone
        df = df.with_columns(
            pl.col("ts")
            .str.strptime(pl.Date, "%Y-%m-%d")
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone("America/New_York")
        )

        return (
            df.select(["ts", "open", "high", "low", "close", "volume", "adj_close"])
            .sort("ts")
        )

    async def search_symbols(self, query: str, market: MarketCode) -> list[dict]:
        url = f"{FMP_BASE}/search"
        params = {
            "query": query,
            "limit": 20,
            "apikey": self._api_key,
        }

        async with self._limiter:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    results = await resp.json()

        return [
            {
                "symbol": r.get("symbol", ""),
                "name": r.get("name", ""),
                "exchange": r.get("stockExchange", ""),
                "sector": r.get("currency", ""),
            }
            for r in results[:20]
        ]

    async def get_universe_symbols(self, universe_name: str, market: MarketCode) -> list[str]:
        if universe_name == "russell2000":
            return RUSSELL2000_SAMPLE

        endpoint_map = {
            "sp500": SP500_ENDPOINT,
            "sp500_liquid": SP500_ENDPOINT,
            "nasdaq100": NASDAQ100_ENDPOINT,
        }

        if universe_name not in endpoint_map:
            raise ValueError(f"Unknown US universe: {universe_name!r}")

        url = f"{FMP_BASE}{endpoint_map[universe_name]}"
        params = {"apikey": self._api_key}

        async with self._limiter:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    constituents = await resp.json()

        symbols = [c.get("symbol", "") for c in constituents if c.get("symbol")]

        # For sp500_liquid, filter by volume if available
        if universe_name == "sp500_liquid":
            # Fetch profile data to filter by avg volume > 500,000
            liquid = []
            for sym_info in constituents:
                sym = sym_info.get("symbol", "")
                if not sym:
                    continue
                # Use the constituent data; fall back to including all
                # FMP constituent endpoint doesn't include volume, so we fetch it separately
                liquid.append(sym)

            # Apply volume filter via batch quote
            if liquid:
                batch = ",".join(liquid[:100])
                quote_url = f"{FMP_BASE}/quote/{batch}"
                async with self._limiter:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(quote_url, params=params) as resp:
                            if resp.status == 200:
                                quotes = await resp.json()
                                symbols = [
                                    q["symbol"] for q in quotes
                                    if q.get("avgVolume", 0) > 500_000
                                ]
                            else:
                                symbols = liquid

        return symbols
