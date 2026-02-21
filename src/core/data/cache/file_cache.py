"""File-based Parquet cache for OHLCV data."""
import time
from datetime import date
from pathlib import Path

import polars as pl


class FileCache:

    def __init__(self, cache_dir: str = "./data/cache"):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, market: str, symbol: str, start: date, end: date, tf: str) -> Path:
        safe = symbol.replace("/", "_").replace(".", "_")
        return self.dir / f"{market}_{safe}_{start}_{end}_{tf}.parquet"

    def get(self, market: str, symbol: str, start: date, end: date, tf: str) -> pl.DataFrame | None:
        p = self._path(market, symbol, start, end, tf)
        return pl.read_parquet(p) if p.exists() else None

    def put(self, df: pl.DataFrame, market: str, symbol: str, start: date, end: date, tf: str) -> None:
        p = self._path(market, symbol, start, end, tf)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(p)

    def is_stale(self, market: str, symbol: str, start: date, end: date, tf: str, max_age_hours: float = 24) -> bool:
        p = self._path(market, symbol, start, end, tf)
        if not p.exists():
            return True
        return (time.time() - p.stat().st_mtime) / 3600 > max_age_hours
