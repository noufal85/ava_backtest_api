"""DataWindow — temporal enforcement. The engine ONLY sees data up to current bar."""
import polars as pl
from dataclasses import dataclass


class TemporalViolationError(Exception):
    pass


@dataclass
class Bar:
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class DataWindow:
    """
    Wraps a DataFrame and enforces strict temporal isolation.
    current_idx is the index of the current bar (bar N).
    The engine may only access bars 0..N via historical() and bar N via current_bar().
    bar N+1 and beyond are NEVER accessible.
    """

    def __init__(self, df: pl.DataFrame, current_idx: int, symbol: str):
        self._df = df
        self._current_idx = current_idx
        self._symbol = symbol

    def current_bar(self) -> pl.DataFrame:
        """Returns bar N only — the bar being processed."""
        return self._df.slice(self._current_idx, 1)

    def historical(self, n: int | None = None) -> pl.DataFrame:
        """Returns bars 0..N-1. Never includes current bar or future bars."""
        end = self._current_idx  # excludes current bar
        data = self._df.slice(0, end)
        if n is not None:
            data = data.tail(n)
        return data

    def indicators(self) -> pl.DataFrame:
        """All indicator values computed up to and including bar N."""
        return self._df.slice(0, self._current_idx + 1)

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def current_idx(self) -> int:
        return self._current_idx

    def __len__(self) -> int:
        return self._current_idx + 1
