"""Shared helper functions for strategy implementations."""
import polars as pl
import numpy as np


def atr(df: pl.DataFrame, period: int) -> pl.Series:
    """Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1).fill_null(0))
    tr3 = abs(low - close.shift(1).fill_null(0))
    tr = pl.max_horizontal(tr1, tr2, tr3)
    return tr.rolling_mean(window_size=period, min_periods=period).alias("atr")


def ema(df: pl.DataFrame, period: int) -> pl.Series:
    """Exponential Moving Average."""
    alpha = 2 / (period + 1)
    return df["close"].ewm_mean(alpha=alpha, adjust=False).alias("ema")


def sma(df: pl.DataFrame, period: int, col: str = "close") -> pl.Series:
    """Simple Moving Average."""
    return df[col].rolling_mean(window_size=period, min_periods=period).alias("sma")


def volume_sma(df: pl.DataFrame, period: int) -> pl.Series:
    """Volume Simple Moving Average."""
    return df["volume"].rolling_mean(window_size=period, min_periods=period).alias("vol_sma")


def bollinger_bands(closes: pl.Series, period: int = 20, std: float = 2.0):
    """Returns (middle, upper, lower) bands."""
    middle = closes.rolling_mean(window_size=period, min_periods=period)
    rolling_std = closes.rolling_std(window_size=period, min_periods=period)
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return {"middle": middle, "upper": upper, "lower": lower}
