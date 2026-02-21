"""Data validation for OHLCV DataFrames."""
from dataclasses import dataclass, field

import polars as pl

from src.core.markets.registry import MarketCode, MARKET_REGISTRY


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


class DataValidator:

    MIN_BARS = 100

    def validate(self, df: pl.DataFrame, market: MarketCode) -> ValidationResult:
        r = ValidationResult()

        if df.is_empty():
            r.errors.append("Empty DataFrame")
            return r

        if len(df) < self.MIN_BARS:
            r.errors.append(f"Too few bars: {len(df)} < {self.MIN_BARS}")

        # OHLC consistency: high must be >= low
        bad = df.filter(pl.col("high") < pl.col("low")).height
        if bad:
            r.errors.append(f"high < low in {bad} rows")

        # No zero or negative prices
        zeros = df.filter((pl.col("close") <= 0) | (pl.col("open") <= 0)).height
        if zeros:
            r.errors.append(f"Zero/negative prices in {zeros} rows")

        # India-specific: tick size check
        if market == MarketCode.IN:
            tick = MARKET_REGISTRY[market].tick_size
            off = df.filter((pl.col("close") % tick).abs() > 0.001).height
            if off:
                r.warnings.append(f"{off} rows off tick size \u20b9{tick}")

        return r
