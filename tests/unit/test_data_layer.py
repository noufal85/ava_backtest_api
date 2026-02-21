"""Unit tests for the EP-2 data provider layer."""
from datetime import date, timedelta

import polars as pl
import pytest

from src.core.data.cache.file_cache import FileCache
from src.core.data.validator.validator import DataValidator, ValidationResult
from src.core.data.providers.nsepy import NSEpyProvider, NIFTY50
from src.core.data.providers.base import DataProvider
from src.core.data.providers.router import ProviderRouter
from src.core.markets.registry import MarketCode


# ── FileCache tests ──────────────────────────────────────────────────────


class TestFileCache:

    def test_put_and_get(self, tmp_path):
        cache = FileCache(cache_dir=str(tmp_path))
        df = pl.DataFrame({
            "ts": [date(2024, 1, 1), date(2024, 1, 2)],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [104.0, 105.0],
            "volume": [1000, 2000],
            "adj_close": [104.0, 105.0],
        })

        cache.put(df, "US", "AAPL", date(2024, 1, 1), date(2024, 1, 2), "1d")
        result = cache.get("US", "AAPL", date(2024, 1, 1), date(2024, 1, 2), "1d")

        assert result is not None
        assert result.shape == df.shape
        assert result["close"].to_list() == [104.0, 105.0]

    def test_get_missing_returns_none(self, tmp_path):
        cache = FileCache(cache_dir=str(tmp_path))
        result = cache.get("US", "AAPL", date(2024, 1, 1), date(2024, 1, 2), "1d")
        assert result is None

    def test_is_stale_missing_file(self, tmp_path):
        cache = FileCache(cache_dir=str(tmp_path))
        assert cache.is_stale("US", "AAPL", date(2024, 1, 1), date(2024, 1, 2), "1d") is True

    def test_is_stale_fresh_file(self, tmp_path):
        cache = FileCache(cache_dir=str(tmp_path))
        df = pl.DataFrame({
            "ts": [date(2024, 1, 1)],
            "close": [100.0],
        })
        cache.put(df, "US", "AAPL", date(2024, 1, 1), date(2024, 1, 1), "1d")
        assert cache.is_stale("US", "AAPL", date(2024, 1, 1), date(2024, 1, 1), "1d", max_age_hours=24) is False

    def test_symbol_with_dots_in_path(self, tmp_path):
        cache = FileCache(cache_dir=str(tmp_path))
        df = pl.DataFrame({"ts": [date(2024, 1, 1)], "close": [100.0]})
        cache.put(df, "IN", "RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 1), "1d")
        result = cache.get("IN", "RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 1), "1d")
        assert result is not None


# ── DataValidator tests ──────────────────────────────────────────────────


class TestDataValidator:

    def _make_valid_df(self, n: int = 200) -> pl.DataFrame:
        """Create a valid OHLCV DataFrame with n rows."""
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        return pl.DataFrame({
            "ts": dates,
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [105.0 + i * 0.1 for i in range(n)],
            "low": [99.0 + i * 0.1 for i in range(n)],
            "close": [104.0 + i * 0.1 for i in range(n)],
            "volume": [1000 + i for i in range(n)],
            "adj_close": [104.0 + i * 0.1 for i in range(n)],
        })

    def test_valid_df_passes(self):
        v = DataValidator()
        df = self._make_valid_df(200)
        result = v.validate(df, MarketCode.US)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_empty_df_fails(self):
        v = DataValidator()
        df = pl.DataFrame({
            "ts": [], "open": [], "high": [], "low": [], "close": [],
            "volume": [], "adj_close": [],
        })
        result = v.validate(df, MarketCode.US)
        assert not result.is_valid
        assert "Empty DataFrame" in result.errors[0]

    def test_too_few_bars_fails(self):
        v = DataValidator()
        df = self._make_valid_df(50)
        result = v.validate(df, MarketCode.US)
        assert not result.is_valid
        assert any("Too few bars" in e for e in result.errors)

    def test_bad_ohlc_high_lt_low(self):
        v = DataValidator()
        df = pl.DataFrame({
            "ts": [date(2020, 1, 1) + timedelta(days=i) for i in range(200)],
            "open": [100.0] * 200,
            "high": [90.0] * 200,   # high < low — invalid
            "low": [95.0] * 200,
            "close": [100.0] * 200,
            "volume": [1000] * 200,
            "adj_close": [100.0] * 200,
        })
        result = v.validate(df, MarketCode.US)
        assert not result.is_valid
        assert any("high < low" in e for e in result.errors)

    def test_zero_prices_fails(self):
        v = DataValidator()
        df = self._make_valid_df(200)
        # Inject a zero close price
        df = df.with_columns(
            pl.when(pl.col("ts") == date(2020, 1, 5))
            .then(pl.lit(0.0))
            .otherwise(pl.col("close"))
            .alias("close")
        )
        result = v.validate(df, MarketCode.US)
        assert not result.is_valid
        assert any("Zero/negative" in e for e in result.errors)

    def test_india_tick_size_warning(self):
        v = DataValidator()
        # Prices not aligned to 0.05 tick
        df = pl.DataFrame({
            "ts": [date(2020, 1, 1) + timedelta(days=i) for i in range(200)],
            "open": [100.03] * 200,
            "high": [105.03] * 200,
            "low": [99.03] * 200,
            "close": [104.03] * 200,
            "volume": [1000] * 200,
            "adj_close": [104.03] * 200,
        })
        result = v.validate(df, MarketCode.IN)
        assert len(result.warnings) > 0
        assert any("tick size" in w for w in result.warnings)


# ── NSEpyProvider tests ──────────────────────────────────────────────────


class TestNSEpyProvider:

    @pytest.mark.asyncio
    async def test_get_universe_symbols_nifty50(self):
        provider = NSEpyProvider()
        symbols = await provider.get_universe_symbols("nifty50", MarketCode.IN)
        assert len(symbols) == len(NIFTY50)
        assert all(s.endswith(".NS") for s in symbols)
        assert "RELIANCE.NS" in symbols

    @pytest.mark.asyncio
    async def test_get_universe_symbols_unknown_raises(self):
        provider = NSEpyProvider()
        with pytest.raises(ValueError, match="Unknown India universe"):
            await provider.get_universe_symbols("ftse100", MarketCode.IN)

    @pytest.mark.asyncio
    async def test_search_symbols(self):
        provider = NSEpyProvider()
        results = await provider.search_symbols("REL", MarketCode.IN)
        assert any(r["symbol"] == "RELIANCE.NS" for r in results)

    def test_supported_markets(self):
        provider = NSEpyProvider()
        assert provider.supported_markets == [MarketCode.IN]
        assert provider.name == "nsepy"


# ── ProviderRouter failover tests ────────────────────────────────────────


class MockProvider(DataProvider):
    """Test helper — a mock DataProvider that can be configured to succeed or fail."""

    def __init__(self, provider_name: str, markets: list[MarketCode], should_fail: bool = False):
        self._name = provider_name
        self._markets = markets
        self._should_fail = should_fail
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_markets(self) -> list[MarketCode]:
        return self._markets

    async def fetch_ohlcv(self, symbol, market, start, end, timeframe="1d"):
        self.call_count += 1
        if self._should_fail:
            raise ConnectionError(f"{self._name} failed")
        return pl.DataFrame({
            "ts": [start],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [104.0],
            "volume": [1000],
            "adj_close": [104.0],
        })

    async def search_symbols(self, query, market):
        return [{"symbol": "TEST", "name": "Test", "exchange": "TEST", "sector": ""}]

    async def get_universe_symbols(self, universe_name, market):
        return ["TEST"]


class TestProviderRouter:

    @pytest.mark.asyncio
    async def test_primary_provider_succeeds(self):
        fmp = MockProvider("fmp", [MarketCode.US], should_fail=False)
        alpaca = MockProvider("alpaca", [MarketCode.US], should_fail=False)
        router = ProviderRouter([fmp, alpaca])

        df = await router.fetch_ohlcv("AAPL", MarketCode.US, date(2024, 1, 1), date(2024, 12, 31))
        assert df is not None
        assert fmp.call_count == 1
        assert alpaca.call_count == 0  # not called because fmp succeeded

    @pytest.mark.asyncio
    async def test_failover_to_secondary(self):
        fmp = MockProvider("fmp", [MarketCode.US], should_fail=True)
        alpaca = MockProvider("alpaca", [MarketCode.US], should_fail=False)
        router = ProviderRouter([fmp, alpaca])

        df = await router.fetch_ohlcv("AAPL", MarketCode.US, date(2024, 1, 1), date(2024, 12, 31))
        assert df is not None
        assert fmp.call_count == 1
        assert alpaca.call_count == 1  # called as fallback

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises(self):
        fmp = MockProvider("fmp", [MarketCode.US], should_fail=True)
        alpaca = MockProvider("alpaca", [MarketCode.US], should_fail=True)
        router = ProviderRouter([fmp, alpaca])

        with pytest.raises(RuntimeError, match="All providers failed"):
            await router.fetch_ohlcv("AAPL", MarketCode.US, date(2024, 1, 1), date(2024, 12, 31))

    @pytest.mark.asyncio
    async def test_india_provider_routing(self):
        nsepy = MockProvider("nsepy", [MarketCode.IN], should_fail=False)
        router = ProviderRouter([nsepy])

        df = await router.fetch_ohlcv("RELIANCE.NS", MarketCode.IN, date(2024, 1, 1), date(2024, 12, 31))
        assert df is not None
        assert nsepy.call_count == 1
