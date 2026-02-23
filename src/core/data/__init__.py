"""Data layer — cache-first provider access.

Design: ALL data access goes through get_provider(), which returns a
CachedProvider wrapping the upstream source.  The Parquet file cache is
checked first; only on a miss (or stale data) does it hit the remote API.
This means:
  • Backtests never hammer the API for data they've already fetched.
  • A single sync pass pre-warms the cache for an entire universe.
  • Switching upstream providers (FMP → Alpaca) is a one-line change.
"""

from src.core.data.cache.file_cache import FileCache
from src.core.data.providers.base import DataProvider
from src.core.data.providers.cached import CachedProvider
from src.core.data.providers.fmp import FMPProvider

# Singleton instances — shared across the process
_cache = FileCache("./data/cache")
_fmp = FMPProvider()
_default_provider: DataProvider = CachedProvider(_fmp, _cache, max_age_hours=24)


def get_provider() -> DataProvider:
    """Return the default cache-first data provider."""
    return _default_provider


def get_cache() -> FileCache:
    """Return the shared file cache instance."""
    return _cache
